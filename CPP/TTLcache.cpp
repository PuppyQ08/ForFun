// ============================================================
// TTLCache — 带过期时间的线程安全内存缓存
//
// 设计目标：
//   1. 任意值类型（模板），key 默认为 std::string
//   2. 读写线程安全（mutex 保护）
//   3. 惰性过期（Lazy Expiry）：只在访问时检查，不起后台线程
//   4. GetOrSet 防击穿（Cache Stampede Protection）：
//      同一 key 并发 miss 时只有一个线程执行 factory，
//      其余线程等待结果，不会同时穿透到下游
// ============================================================

#include <unordered_map>   // 存储 key->Entry 的哈希表
#include <string>          // 默认 Key 类型
#include <chrono>          // 时间点、时长，用于 TTL 计算
#include <shared_mutex>    // std::shared_mutex / shared_lock（读锁）/ unique_lock（写锁）
#include <folly/futures/Future.h>         // folly::SemiFuture / folly::makeSemiFuture
#include <folly/futures/SharedPromise.h>  // folly::SharedPromise：一写多读，每个 follower 独立 future
#include <memory>          // std::shared_ptr / std::make_shared
#include <stdexcept>       // 异常基类（factory 可能抛出）

// ──────────────────────────────────────────────
// 模板参数设计：
//   Value     — 缓存存储的值类型，由调用方决定（唯一必须指定的参数）
//   Key       — 键类型，默认 std::string；换成 int/UUID 等只需改此参数
//   Clock     — 时钟策略，默认 steady_clock（单调递增，不受系统时间调整影响）
//               可换成 system_clock（可与外部时间戳比较）或 mock 时钟（方便测试）
//   Hash      — 哈希函数，和 std::unordered_map 保持一致的扩展点
//   KeyEqual  — 键相等比较，支持自定义（如大小写不敏感的字符串键）
//
// C++ 特性：模板类 + 默认模板参数
// ──────────────────────────────────────────────
template <
    class Value,
    class Key      = std::string,
    class Clock    = std::chrono::steady_clock,
    class Hash     = std::hash<Key>,
    class KeyEqual = std::equal_to<Key>
>
class TTLCache {
public:
    // ── 类型别名（using）：让内部代码更简洁，也方便外部通过 TTLCache<V>::ptr_type 使用 ──
    using value_type = Value;
    using key_type   = Key;
    using clock_type = Clock;
    using time_point = typename Clock::time_point;   // 时钟对应的时间点类型
    using duration   = std::chrono::milliseconds;    // 统一用毫秒表示 TTL，避免单位混乱

    // ptr_type = shared_ptr<const Value>
    // 设计原因：
    //   · shared_ptr  — 多个调用方可共享同一份 Value，避免深拷贝
    //   · const Value — 缓存中的值不应被外部修改；如需更新应调用 Put
    using ptr_type    = std::shared_ptr<const Value>;

    // future_type：GetOrSet 的异步返回类型
    // SemiFuture 是"无 executor 的 future"，调用方通过 .via(executor).thenValue(...)
    // 绑定执行线程后再链式处理，不会阻塞当前线程
    using future_type = folly::SemiFuture<ptr_type>;

    // promise_type：leader 用来向所有 follower 广播结果
    // SharedPromise 可以多次调用 getSemiFuture()，每次返回独立的 future 对象，
    // 所有这些 future 在 setValue/setException 时同时就绪
    // （对比 std::promise：只能 get_future() 一次）
    using promise_type = folly::SharedPromise<ptr_type>;

    // ──────────────────────────────────────────
    // 构造函数
    // explicit 防止隐式转换（如 TTLCache<int> c = 500 这种误用）
    // default_ttl = 0  表示"永不过期"（见 MakeExpireAt_ 的实现）
    // ──────────────────────────────────────────
    explicit TTLCache(duration default_ttl = duration{0})
        : default_ttl_(default_ttl) {}

    // ══════════════════════════════════════════════════════════════
    // Get — 读取缓存
    //
    // 锁策略（两阶段）：
    //   阶段1：shared_lock（读锁）— 多线程并发读，互不阻塞
    //     · 命中且未过期 → 持读锁直接返回，性能最优
    //     · key 不存在   → 直接返回 false
    //     · 已过期       → 释放读锁，进入阶段2 用写锁清理
    //   阶段2：unique_lock（写锁）+ 双重检查
    //     · 阶段1 释放读锁到阶段2 拿到写锁之间存在真实并发窗口：
    //       另一线程可能已经 Put 了新值，必须再次确认仍过期才能 erase
    // ══════════════════════════════════════════════════════════════
    bool Get(const key_type& key, ptr_type& out) {
        // ── 阶段1：共享读锁，命中直接返回 ──
        {
            std::shared_lock<std::shared_mutex> lk(mu_);
            const auto now = Clock::now();
            auto it = map_.find(key);
            if (it == map_.end()) return false;   // key 不存在

            if (!IsExpiredUnlocked(it->second, now)) {
                out = it->second.value;   // 命中且未过期，shared_ptr 引用计数 +1
                return true;
            }
            // 已过期，需要写锁才能 erase，先释放读锁
        }

        // ── 阶段2：独占写锁 + 双重检查，清理过期条目 ──
        {
            std::unique_lock<std::shared_mutex> lk(mu_);
            const auto now = Clock::now();
            auto it = map_.find(key);
            // 双重检查：阶段1 到阶段2 之间另一线程可能已 Put 了新值
            // 只有确认仍过期才 erase，避免误删刚写入的有效值
            if (it != map_.end() && IsExpiredUnlocked(it->second, now)) {
                map_.erase(it);
            }
        }
        return false;
    }

    // ══════════════════════════════════════════════════════════════
    // Put — 写入缓存
    //
    // ttl = -1（哨兵值）表示"使用构造时设定的 default_ttl_"
    // ttl =  0 表示"永不过期"
    // ttl >  0 表示具体的过期时长
    //
    // Value v 按值传入（而非 const ref），是为了支持 std::move：
    //   · 调用方可以传入临时对象，直接移动进来，避免一次拷贝
    //   · std::make_shared 内部再移动一次构造堆上对象
    // ══════════════════════════════════════════════════════════════
    void Put(const key_type& key, Value v, duration ttl = duration{-1}) {
        if (ttl.count() < 0) ttl = default_ttl_;   // 负数哨兵 -> 使用默认 TTL
        const auto expire_at = MakeExpireAt_(ttl, Clock::now());

        // make_shared 在锁外执行：构造对象不需要持锁，减少锁持有时间
        auto p = std::make_shared<Value>(std::move(v));

        std::unique_lock<std::shared_mutex> lk(mu_);
        // map_[key] = ... 若 key 已存在则覆盖（更新语义）
        map_[key] = Entry{std::move(p), expire_at};
    }

    // ══════════════════════════════════════════════════════════════
    // GetOrSet — 缓存的核心接口，兼具读取和惰性填充（异步版本）
    //
    // 返回 folly::SemiFuture<ptr_type>，而非直接返回 ptr_type：
    //   · 命中缓存    → 返回已就绪的 future（makeSemiFuture），无阻塞
    //   · follower   → 返回 SharedPromise::getSemiFuture()，不阻塞线程，
    //                  factory 完成后由调用方绑定的 executor 调度回调
    //   · leader     → 执行 factory，写回缓存，通过 SharedPromise::setValue
    //                  同时唤醒所有 follower 的 future
    //
    // 解决线程池耗尽问题：
    //   std 版本：follower 调 .get() 挂起线程，factory 慢时线程池全部阻塞
    //   Folly 版本：follower 拿到 SemiFuture 立即返回，线程不挂起，
    //   调用方通过 .via(executor).thenValue([](ptr_type p){ ... }) 注册回调，
    //   factory 完成后 executor 调度执行，线程池始终可以处理其他请求
    //
    // 锁策略与 DCL：同前，shared_lock（阶段1）→ unique_lock（阶段2）
    // ══════════════════════════════════════════════════════════════
    template <class Factory>
    future_type GetOrSet(const key_type& key, Factory&& factory, duration ttl = duration{-1}) {
        if (ttl.count() < 0) ttl = default_ttl_;

        // ── 阶段1：共享读锁，乐观快速路径 ──
        // 多个线程可同时执行此段，高命中率场景下并发性能最优
        {
            std::shared_lock<std::shared_mutex> lk(mu_);
            const auto now = Clock::now();
            auto it = map_.find(key);
            if (it != map_.end() && !IsExpiredUnlocked(it->second, now)) {
                // 命中：用 makeSemiFuture 包装成已就绪的 future，调用方可直接 .value()
                return folly::makeSemiFuture(it->second.value);
            }
            // 未命中或已过期，释放读锁进入阶段2
            // ← 读锁在此释放，持写锁的线程现在可以写入（真实并发窗口）
        }

        // ── 阶段2：独占写锁，双重检查 + inflight 去重 ──
        std::shared_ptr<promise_type> sp;   // SharedPromise：leader 创建，follower 共享
        bool i_am_leader = false;

        {
            std::unique_lock<std::shared_mutex> lk(mu_);

            // 双重检查：阶段1 释放读锁到此处拿到写锁之间，
            // 另一线程可能已完成写入，此时直接命中返回，不需要执行 factory
            const auto now = Clock::now();
            auto it = map_.find(key);
            if (it != map_.end() && !IsExpiredUnlocked(it->second, now)) {
                return folly::makeSemiFuture(it->second.value);
            }
            if (it != map_.end()) {
                map_.erase(it);   // 确认已过期，清理
            }

            // 检查是否已有其他线程正在构造这个 key
            auto inflight_it = inflight_.find(key);
            if (inflight_it != inflight_.end()) {
                // follower：从已有的 SharedPromise 签出一个独立 SemiFuture
                // getSemiFuture() 每次返回新对象，多个 follower 互不干扰，
                // 且不阻塞当前线程——future 在 setValue 后由 executor 调度
                sp = inflight_it->second;
            } else {
                // leader：创建 SharedPromise，登记到 inflight_
                // SharedPromise 允许多次 getSemiFuture()（std::promise 只能一次）
                sp = std::make_shared<promise_type>();
                inflight_[key] = sp;
                i_am_leader = true;
            }
        }

        if (!i_am_leader) {
            // follower：非阻塞返回，线程立即释放回线程池
            // 调用方通过 .via(executor).thenValue(...) 注册回调，
            // leader 的 sp->setValue() 触发后由 executor 调度执行
            return sp->getSemiFuture();
        }

        // ── 阶段3：leader 在锁外执行 factory（耗时操作不持锁，不阻塞其他 key）──
        ptr_type built;
        try {
            Value v = factory();   // 可能很慢（查 DB、调 RPC）
            built = std::make_shared<Value>(std::move(v));
        } catch (...) {
            // factory 抛异常：
            //   1. 持写锁清理 inflight_
            //   2. setException 把异常传播给所有 follower 的 future
            //      调用方在 .thenValue 链上的 .thenError 或 .via().thenError 中处理
            //   3. 重新抛出，leader 调用方同样感知异常
            {
                std::unique_lock<std::shared_mutex> lk(mu_);
                inflight_.erase(key);
            }
            sp->setException(std::current_exception());
            throw;
        }

        // ── 阶段4：leader 将结果写回缓存，广播给所有 follower ──
        {
            std::unique_lock<std::shared_mutex> lk(mu_);
            const auto now = Clock::now();
            map_[key] = Entry{built, MakeExpireAt_(ttl, now)};
            inflight_.erase(key);
        }
        // setValue 在锁外调用：避免 follower 回调被调度时立即竞争写锁
        // 所有通过 getSemiFuture() 拿到 future 的 follower 此刻同时就绪
        sp->setValue(built);
        return folly::makeSemiFuture(built);
    }

    // ══════════════════════════════════════════════════════════════
    // Clear — 清空整个缓存（包括正在进行中的 inflight 任务）
    // ══════════════════════════════════════════════════════════════
    void Clear() {
        std::unique_lock<std::shared_mutex> lk(mu_);
        map_.clear();
        inflight_.clear();
    }

    // ══════════════════════════════════════════════════════════════
    // CleanupExpired — 主动扫描并删除所有已过期条目
    //
    // 惰性过期的补充：适合在低峰期（如定时任务、每N次操作后）调用，
    // 控制内存上限，防止大量冷 key 积压。
    // 返回删除的条目数量（供监控/日志使用）。
    //
    // 迭代器安全：unordered_map::erase(it) 返回下一个有效迭代器（C++11 起）
    // ══════════════════════════════════════════════════════════════
    std::size_t CleanupExpired() {
        std::unique_lock<std::shared_mutex> lk(mu_);
        const auto now = Clock::now();
        std::size_t removed = 0;
        for (auto it = map_.begin(); it != map_.end();) {
            if (IsExpiredUnlocked(it->second, now)) {
                it = map_.erase(it);   // erase 返回下一个迭代器，避免迭代器失效
                ++removed;
            } else {
                ++it;
            }
        }
        return removed;
    }

private:
    // ══════════════════════════════════════════════════════════════
    // Erase — 主动删除单个 key（设为 private）
    //
    // 不对外暴露的原因：
    //   TTL cache 的核心契约是"时间到自动失效"，手动删除会破坏这一语义。
    //   更严重的是，Erase 无法中断正在进行中的 GetOrSet：
    //   若 leader 正在锁外执行 factory()，Erase 完成后 leader 仍会
    //   在阶段4 将结果写回 map_，导致 Erase 被静默撤销，调用方毫不知情。
    //   内部（如未来扩展 LRU 淘汰）可以安全使用，外部调用方应依赖 TTL。
    // ══════════════════════════════════════════════════════════════
    void Erase(const key_type& key) {
        std::unique_lock<std::shared_mutex> lk(mu_);
        map_.erase(key);
        inflight_.erase(key);
    }

    // ──────────────────────────────────────────
    // Entry：缓存中每个 key 对应的存储单元
    //   value     — 指向堆上值的 shared_ptr（const，外部不可修改）
    //   expire_at — 过期时间点；time_point::max() 表示永不过期
    // ──────────────────────────────────────────
    struct Entry {
        ptr_type   value;
        time_point expire_at;
    };

    // ──────────────────────────────────────────
    // MakeExpireAt_：将 TTL 时长转换为绝对时间点
    //
    // ttl == 0 -> time_point::max()（永不过期的哨兵值）
    //          用 max() 而非特殊 bool 字段，避免 Entry 结构增加字段，
    //          且与 IsExpiredUnlocked 的判断逻辑自然配合
    // ──────────────────────────────────────────
    static time_point MakeExpireAt_(duration ttl, time_point now) {
        if (ttl.count() == 0) return time_point::max();
        return now + ttl;
    }

    // ──────────────────────────────────────────
    // IsExpiredUnlocked：判断条目是否已过期
    //
    // "Unlocked" 后缀是一种命名惯例：提醒调用方此函数不加锁，
    // 必须在已持有 mu_ 的情况下调用（锁内调用锁内函数，避免重复加锁死锁）
    //
    // 逻辑：expire_at == max() 表示永不过期，短路返回 false；
    //       否则判断当前时间是否已超过过期时间点
    // ──────────────────────────────────────────
    static bool IsExpiredUnlocked(const Entry& e, time_point now) {
        return e.expire_at != time_point::max() && now >= e.expire_at;
    }

private:
    // ──────────────────────────────────────────
    // 成员变量
    //
    // mu_          — 保护 map_ 和 inflight_ 的互斥锁（二者总是一起访问）
    // map_         — 主存储：key -> Entry（value + expire_at）
    // inflight_    — 并发防击穿：key -> shared_future（正在构造中的 key）
    // default_ttl_ — 构造时设定的默认过期时长
    //
    // Hash/KeyEqual 模板参数透传给 unordered_map，支持自定义哈希和比较策略
    // ──────────────────────────────────────────
    // shared_mutex：读操作用 shared_lock（读读并发），写操作用 unique_lock（独占）
    std::shared_mutex mu_;
    std::unordered_map<key_type, Entry,                             Hash, KeyEqual> map_;
    // inflight_ 存 SharedPromise 而非 shared_future：
    //   · SharedPromise 支持多次 getSemiFuture()，每个 follower 拿独立 future
    //   · 持有 shared_ptr 保证 leader 在锁外 setValue 时对象仍然存活
    std::unordered_map<key_type, std::shared_ptr<promise_type>,    Hash, KeyEqual> inflight_;
    duration default_ttl_;
};

// ══════════════════════════════════════════════════════════════
// 使用示例
// ══════════════════════════════════════════════════════════════

// 创建缓存：Value=int，Key=string（默认），默认 TTL=500ms
TTLCache<int> c(std::chrono::milliseconds(500));

folly::Executor* ex = /* 业务线程池，如 folly::CPUThreadPoolExecutor */ nullptr;

// miss：返回 SemiFuture，factory 执行后 future 就绪
// .via(ex) 绑定 executor，.thenValue 注册回调，当前线程不阻塞
c.GetOrSet("answer", [] { return 42; })
    .via(ex)
    .thenValue([](TTLCache<int>::ptr_type p) {
        // factory 完成后由 ex 调度执行，线程池不被占用
        use(*p);
    });

// hit：SemiFuture 已就绪（makeSemiFuture），.thenValue 几乎立即执行
c.GetOrSet("answer", [] { return 43; })
    .via(ex)
    .thenValue([](TTLCache<int>::ptr_type p) { use(*p); });

// 自定义 TTL：2s 过期
c.GetOrSet("k", [] { return 7; }, std::chrono::seconds(2))
    .via(ex)
    .thenValue([](TTLCache<int>::ptr_type p) { use(*p); })
    .thenError(folly::tag_t<std::exception>{}, [](const std::exception& e) {
        // factory 抛出的异常在这里处理，不会 crash follower 线程
    });

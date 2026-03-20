#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <chrono>
#include <mutex>
#include <string>

namespace mccl {

enum class LogLevel : int {
    TRACE = 0,
    DEBUG = 1,
    INFO  = 2,
    WARN  = 3,
    ERROR = 4,
    FATAL = 5,
    OFF   = 6,
};

inline LogLevel log_level_from_env() {
    const char* env = std::getenv("MCCL_LOG_LEVEL");
    if (!env) return LogLevel::WARN;
    std::string s(env);
    if (s == "TRACE" || s == "trace" || s == "0") return LogLevel::TRACE;
    if (s == "DEBUG" || s == "debug" || s == "1") return LogLevel::DEBUG;
    if (s == "INFO"  || s == "info"  || s == "2") return LogLevel::INFO;
    if (s == "WARN"  || s == "warn"  || s == "3") return LogLevel::WARN;
    if (s == "ERROR" || s == "error" || s == "4") return LogLevel::ERROR;
    if (s == "FATAL" || s == "fatal" || s == "5") return LogLevel::FATAL;
    if (s == "OFF"   || s == "off"   || s == "6") return LogLevel::OFF;
    return LogLevel::WARN;
}

inline LogLevel& global_log_level() {
    static LogLevel level = log_level_from_env();
    return level;
}

inline void refresh_log_level() {
    global_log_level() = log_level_from_env();
}

inline const char* level_str(LogLevel l) {
    switch (l) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "?????";
    }
}

inline void log_impl(LogLevel level, const char* file, int line,
                      const char* fmt, ...) {
    if (level < global_log_level()) return;

    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);

    auto now = std::chrono::system_clock::now();
    auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    fprintf(stderr, "[MCCL %s %lld %s:%d] ",
            level_str(level), static_cast<long long>(epoch_ms), file, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");
    fflush(stderr);
}

#define MCCL_LOG(lvl, ...)                                             \
    ::mccl::log_impl(::mccl::LogLevel::lvl, __FILE__, __LINE__, __VA_ARGS__)

#define MCCL_TRACE(...) MCCL_LOG(TRACE, __VA_ARGS__)
#define MCCL_DEBUG(...) MCCL_LOG(DEBUG, __VA_ARGS__)
#define MCCL_INFO(...)  MCCL_LOG(INFO,  __VA_ARGS__)
#define MCCL_WARN(...)  MCCL_LOG(WARN,  __VA_ARGS__)
#define MCCL_ERROR(...) MCCL_LOG(ERROR, __VA_ARGS__)

// MCCL_FATAL logs at the highest severity level but does NOT terminate the
// process.  Use it for serious errors that are being handled gracefully (e.g.
// a collective that will surface as a Python exception via markError).
#define MCCL_FATAL(...) MCCL_LOG(FATAL, __VA_ARGS__)

// MCCL_ABORT logs at FATAL level and then terminates the process immediately.
// Reserve this for truly unrecoverable conditions where continuing would
// corrupt state (e.g. an assertion failure inside a callback that has no
// error-propagation path).
#define MCCL_ABORT(...)                                                \
    do {                                                               \
        MCCL_LOG(FATAL, __VA_ARGS__);                                  \
        std::terminate();                                              \
    } while (false)

} // namespace mccl

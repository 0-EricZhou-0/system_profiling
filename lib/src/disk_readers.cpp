#include "disk_readers.h"

#include <cerrno>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>

namespace cupti_profiler {
namespace internal {

std::vector<DiskStatSnapshot> ReadDiskStats(const std::vector<std::string>& devices) {
    std::vector<DiskStatSnapshot> results;
    std::ifstream f("/proc/diskstats");
    if (!f) return results;

    std::string line;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        uint64_t major, minor;
        std::string devName;
        iss >> major >> minor >> devName;

        // Check if this device is in our list
        bool wanted = false;
        for (const auto& d : devices) {
            if (d == devName) { wanted = true; break; }
        }
        if (!wanted) continue;

        // Fields after device name (1-indexed from the kernel docs):
        //  1: reads completed
        //  2: reads merged
        //  3: sectors read      ← we want this (field index 3, 0-based field[2])
        //  4: read time ms
        //  5: writes completed
        //  6: writes merged
        //  7: sectors written   ← we want this (field index 7, 0-based field[6])
        //  ...
        std::vector<uint64_t> fields;
        uint64_t val;
        while (iss >> val) {
            fields.push_back(val);
        }

        DiskStatSnapshot snap;
        snap.device = devName;
        if (fields.size() > 2) snap.sectorsRead = fields[2];
        if (fields.size() > 6) snap.sectorsWritten = fields[6];
        results.push_back(snap);
    }
    return results;
}

DiskInflightSnapshot ReadDiskInflight(const std::string& device) {
    DiskInflightSnapshot s;
    std::string path = "/sys/block/" + device + "/inflight";
    std::ifstream f(path);
    if (!f) return s;
    f >> s.readInflight >> s.writeInflight;
    return s;
}

PIDIOSnapshot ReadPIDIO(uint32_t pid) {
    PIDIOSnapshot s;
    std::string path = "/proc/" + std::to_string(pid) + "/io";
    // open/read directly, for the errno: the kernel checks access both
    // at open (file mode, owner) and at read (ptrace access).
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        s.accessible = false;
        s.error = errno;
        return s;
    }
    std::string text;
    char buf[512];
    for (;;) {
        ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) { s.error = errno; break; }
        if (n == 0) break;
        text.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);
    if (s.error) {
        s.accessible = false;
        return s;
    }
    std::istringstream f(text);

    // Format: "key: value", one per line.
    struct Field { const char* key; size_t len; uint64_t PIDIOSnapshot::* dst; };
    static constexpr Field kFields[] = {
        {"rchar: ",                 7, &PIDIOSnapshot::rchar},
        {"wchar: ",                 7, &PIDIOSnapshot::wchar},
        {"read_bytes: ",           12, &PIDIOSnapshot::readBytes},
        {"write_bytes: ",          13, &PIDIOSnapshot::writeBytes},
        {"cancelled_write_bytes: ", 23, &PIDIOSnapshot::cancelledWriteBytes},
    };
    std::string line;
    int found = 0;
    while (std::getline(f, line)) {
        for (const auto& k : kFields) {
            if (line.compare(0, k.len, k.key) == 0) {
                std::istringstream(line.substr(k.len)) >> s.*k.dst;
                ++found;
                break;
            }
        }
    }
    if (found < 5) {   // never zeros for a reading
        s.accessible = false;
        s.error = ENODATA;
    }
    return s;
}

} // namespace internal
} // namespace cupti_profiler

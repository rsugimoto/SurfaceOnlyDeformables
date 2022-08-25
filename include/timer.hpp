#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

class Timer {
  public:
    Timer(std::string stat_save_path = "") : stat_save_path(stat_save_path) {}

    inline void set_stat_save_path(std::string stat_save_path) { this->stat_save_path = stat_save_path; }

    template <class T> inline auto measure(std::string name, T lambda) {
        auto t_begin = std::chrono::high_resolution_clock::now();
        lambda();
        auto t_end = std::chrono::high_resolution_clock::now();
        auto t_diff = t_end - t_begin;
        if (duration.count(name) == 0) duration[name] = std::vector<std::chrono::duration<int64_t, std::nano>>();
        duration[name].push_back(t_diff);

        return t_diff;
    }

    template <class T = std::chrono::nanoseconds> inline void display_average_times() {
        std::cout << "-----Average Execution Time-----" << std::endl;

        for (const auto &[key, duration_vec] : duration) {
            std::chrono::duration<int64_t, std::nano> duration_sum(0);
            for (const auto _duration : duration_vec) duration_sum += _duration;
            std::cout << key << ": " << std::chrono::duration_cast<T>(duration_sum / duration_vec.size()).count()
                      << std::endl;
        }

        std::cout << "--------------------------------" << std::endl;

        if (stat_save_path != "") {
            if (stat_save_path.parent_path() != "") std::filesystem::create_directories(stat_save_path.parent_path());
            std::ofstream file;
            file.open(stat_save_path);
            for (const auto &[key, duration_vec] : duration) {
                file << "\"" << key << "\",";
                for (size_t i = 0; i < duration_vec.size(); i++) {
                    file << std::chrono::duration_cast<T>(duration_vec[i]).count();
                    if (i != duration_vec.size() - 1) file << ",";
                }
                file << std::endl;
            }
            file.close();
        }
    }

  private:
    std::map<std::string, std::vector<std::chrono::duration<int64_t, std::nano>>> duration;
    std::filesystem::path stat_save_path;
};

#endif //!__TIMER_HPP__
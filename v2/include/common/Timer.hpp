#include <chrono>
#include <iostream>

class Timer {
public:
    Timer() : paused(true), elapsed_time(0) {}

    void start() {
        if (paused) {
            start_time = std::chrono::high_resolution_clock::now();
            paused = false;
        }
    }

    void pause() {
        if (!paused) {
            auto current_time = std::chrono::high_resolution_clock::now();
            elapsed_time += std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
            paused = true;
        }
    }

    void resume() {
        start(); // Resume works just like starting the timer, but without resetting elapsed time
    }

    void reset() {
        paused = true;
        elapsed_time = 0;
    }

    long long getElapsedMilliseconds() {
        if (!paused) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto current_elapsed_time = elapsed_time + std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
            return current_elapsed_time;
        }
        return elapsed_time;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    bool paused;
    long long elapsed_time; // milliseconds
};

// int main() {
//     Timer timer;
//     timer.start();
//     std::cout << "Timer started" << std::endl;

//     // Simulate some work
//     std::this_thread::sleep_for(std::chrono::seconds(2));
//     std::cout << "2 seconds elapsed" << std::endl;

//     timer.pause();
//     std::cout << "Timer paused. Elapsed time: " << timer.getElapsedMilliseconds() << " ms" << std::endl;

//     // Simulate pause
//     std::this_thread::sleep_for(std::chrono::seconds(1));

//     timer.resume();
//     std::cout << "Timer resumed" << std::endl;

//     // Simulate more work
//     std::this_thread::sleep_for(std::chrono::seconds(2));
//     std::cout << "2 more seconds elapsed" << std::endl;

//     timer.pause();
//     std::cout << "Timer paused again. Total elapsed time: " << timer.getElapsedMilliseconds() << " ms" << std::endl;

//     return 0;
// }

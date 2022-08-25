#include "simulator.hpp"
#include "timer.hpp"

int main(int argc, char *argv[]) {
    Simulator simulator;
    if (!simulator.init(argc, argv)) return 1;
    simulator.loop();
    return 0;
}
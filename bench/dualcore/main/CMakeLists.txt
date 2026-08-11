# The dot kernel is shared with bench/vector rather than copied: the two benchmarks
# must measure the same instruction sequence for their numbers to be comparable.
idf_component_register(SRCS "mem_main.c" "../../vector/main/pie_dot.S"
                       REQUIRES esp_psram esp_timer esp_hw_support)

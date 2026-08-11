# The engine's own row kernel, not a copy of it: the point of this benchmark is
# to explain the engine's measured cost, which only works if it runs the same
# instructions.
idf_component_register(SRCS "gemm_main.c" "../../../engine/pie_rows.S"
                            "../../../engine/pie_attn.S"
                            "qacc_probe.S"
                            "pie_qk8.S"
                       REQUIRES esp_psram esp_timer)

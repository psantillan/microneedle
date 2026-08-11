# Gzip the browser assets and embed them in the app image. 163 KB of source
# becomes ~72 KB of flash, against 1.75 MiB spare in the app partition.
#
# The gzip runs as a build-time custom command with a real DEPENDS, not
# execute_process: gzipping at configure time meant editing the page did not
# rebuild the image and the board silently kept serving the previous copy.
# The commands are guarded because IDF first evaluates this file in script
# mode to collect requirements, where add_custom_command is not allowed.
# The web UI is optional: CONFIG_NEEDLE_WEB_UI=n builds an API-only image and
# skips embedding the page, its tokenizer and the vocabulary.
set(WEB_SRC
    "${CMAKE_CURRENT_LIST_DIR}/../../web/index.html"
    "${CMAKE_CURRENT_LIST_DIR}/../../web/bpe.js"
    "${CMAKE_CURRENT_LIST_DIR}/../../web/needle.js"
    "${CMAKE_CURRENT_LIST_DIR}/../../web/tool_retrieval.js"
    "${CMAKE_CURRENT_LIST_DIR}/../../tokenizer/vocab.txt")

set(WEB_GZ "")
foreach(f ${WEB_SRC})
    get_filename_component(n "${f}" NAME)
    list(APPEND WEB_GZ "${CMAKE_CURRENT_BINARY_DIR}/web/${n}.gz")
endforeach()

if(NOT CMAKE_BUILD_EARLY_EXPANSION AND CONFIG_NEEDLE_WEB_UI)
    foreach(f ${WEB_SRC})
        get_filename_component(n "${f}" NAME)
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/web/${n}.gz"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/web"
            COMMAND gzip -9 -c "${f}" > "${CMAKE_CURRENT_BINARY_DIR}/web/${n}.gz"
            DEPENDS "${f}"
            COMMENT "gzip ${n}"
            VERBATIM)
    endforeach()
    add_custom_target(web_assets DEPENDS ${WEB_GZ})
endif()

if(NOT CONFIG_NEEDLE_WEB_UI)
    set(WEB_GZ "")
endif()

idf_component_register(SRCS "fw_main.c" "net.c"
                            "../../engine/needle_engine.c"
                            "../../engine/pie_rows.S"
                            "../../engine/pie_attn.S"
                       INCLUDE_DIRS "." "../../engine"
                       EMBED_FILES ${WEB_GZ}
                       REQUIRES esp_psram esp_partition esp_timer esp_driver_uart vfs esp_app_format
                                esp_eth esp_netif esp_event esp_http_server)
target_compile_definitions(${COMPONENT_LIB} PRIVATE NE_PIE NE_ATTN_I8 NE_WARMSCHEMA)
if(NOT CMAKE_BUILD_EARLY_EXPANSION AND CONFIG_NEEDLE_WEB_UI)
    add_dependencies(${COMPONENT_LIB} web_assets)
endif()

/* Closed loop: the board serves its own demo page over Ethernet.
 *
 * The tokenizer runs in the browser, so the board hands out 72 KB of gzipped
 * static assets and answers token ids. Nothing else is on the network path.
 *
 * The PIE context-save bug in IDF 5.4.2 forces a scheduler guard around vector
 * work. That guard is taken per burst (see ne_critical_enter in
 * needle_engine.h), not per query: a whole query still holds the CPU for the
 * full prefill (board-certified ~5.9 s at 271 tokens) plus decode, which no
 * TCP connection survives as one critical section, while per-burst hands lwIP
 * the CPU back every few milliseconds (longest single window ~40 ms when the
 * logits GEMV runs).
 *
 * Hardware: Waveshare ESP32-P4-WIFI6-POE-ETH, IP101GRI PHY at address 1,
 * RMII with an external 50 MHz clock. Every RMII pin on this board matches
 * ETH_ESP32_EMAC_DEFAULT_CONFIG() for the P4, so only the reset pin and the
 * PHY address are board-specific.
 */

#include <string.h>

#include "esp_eth.h"
#include "esp_eth_mac_esp.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_mac.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_app_desc.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#include "needle_engine.h"
#ifdef NE_WARMSCHEMA
extern char g_infer_path;
#endif
#include "net.h"

static const char *TAG = "net";

#define PHY_ADDR      1
#define PHY_RESET_IO  51

#if CONFIG_NEEDLE_WEB_UI
/* gzipped at build time by CMake; see main/CMakeLists.txt */
extern const uint8_t index_html_start[] asm("_binary_index_html_gz_start");
extern const uint8_t index_html_end[]   asm("_binary_index_html_gz_end");
extern const uint8_t bpe_js_start[]     asm("_binary_bpe_js_gz_start");
extern const uint8_t bpe_js_end[]       asm("_binary_bpe_js_gz_end");
extern const uint8_t needle_js_start[]  asm("_binary_needle_js_gz_start");
extern const uint8_t needle_js_end[]    asm("_binary_needle_js_gz_end");
extern const uint8_t retr_js_start[]    asm("_binary_tool_retrieval_js_gz_start");
extern const uint8_t retr_js_end[]      asm("_binary_tool_retrieval_js_gz_end");
extern const uint8_t vocab_start[]      asm("_binary_vocab_txt_gz_start");
extern const uint8_t vocab_end[]        asm("_binary_vocab_txt_gz_end");

#endif /* CONFIG_NEEDLE_WEB_UI */

static char g_ip[16] = "0.0.0.0";
/* The engine holds one model in file-scope state and the serial console can
 * start a query too. Two concurrent callers would interleave inside ne_encode
 * and silently corrupt each other, so a request that arrives mid-inference is
 * refused rather than queued -- a multi-second queue behind a multi-second
 * request is worse than an honest 503. */
static SemaphoreHandle_t g_busy;

/* Created before anything can serve a request, including the serial console,
 * which is why this is not inside net_start(). */
__attribute__((constructor)) static void net_busy_init(void)
{ g_busy = xSemaphoreCreateMutex(); }
static ne_infer_fn g_infer;

const char *net_ip(void) { return g_ip; }

int  net_infer_try_lock(void)
{ return g_busy && xSemaphoreTake(g_busy, 0) == pdTRUE; }
void net_infer_unlock(void) { if (g_busy) xSemaphoreGive(g_busy); }

#if CONFIG_NEEDLE_WEB_UI
/* ---------- static assets ---------- */

static esp_err_t send_gz(httpd_req_t *r, const uint8_t *a, const uint8_t *b,
                         const char *type)
{
    httpd_resp_set_type(r, type);
    httpd_resp_set_hdr(r, "Content-Encoding", "gzip");
    httpd_resp_set_hdr(r, "Cache-Control", "no-cache");
    return httpd_resp_send(r, (const char *)a, b - a);
}

static esp_err_t h_index(httpd_req_t *r)
{ return send_gz(r, index_html_start, index_html_end, "text/html; charset=utf-8"); }

static esp_err_t h_bpe(httpd_req_t *r)
{ return send_gz(r, bpe_js_start, bpe_js_end, "text/javascript; charset=utf-8"); }

static esp_err_t h_needle(httpd_req_t *r)
{ return send_gz(r, needle_js_start, needle_js_end, "text/javascript; charset=utf-8"); }

static esp_err_t h_retr(httpd_req_t *r)
{ return send_gz(r, retr_js_start, retr_js_end, "text/javascript; charset=utf-8"); }

static esp_err_t h_vocab(httpd_req_t *r)
{ return send_gz(r, vocab_start, vocab_end, "text/plain; charset=utf-8"); }

#endif /* CONFIG_NEEDLE_WEB_UI */

/* ---------- status ---------- */

static esp_err_t h_status(httpd_req_t *r)
{
    /* Enough for the board to answer questions about itself -- it is the one
     * tool that needs no network and cannot go stale. */
    const esp_app_desc_t *app = esp_app_get_description();
    char buf[640];
    int n = snprintf(buf, sizeof buf,
                     "{\"ready\":%s,\"selftest_failed\":%s,\"port\":\"onboard\","
                     "\"banner\":[%s],\"ip\":\"%s\","
                     "\"uptime_s\":%llu,\"psram_free_kb\":%u,\"internal_free_kb\":%u,"
                     "\"cpu_mhz\":%d,\"idf\":\"%s\",\"built\":\"%s %s\","
                     "\"model\":\"Needle 26M\",\"weights_mib\":14.66}",
                     net_selftest_failed() ? "false" : "true",
                     net_selftest_failed() ? "true" : "false",
                     net_selftest_json(), g_ip,
                     (unsigned long long)(esp_timer_get_time() / 1000000ULL),
                     (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
                     (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
                     360, app->idf_ver, app->date, app->time);
    if (n > (int)sizeof buf - 1) n = (int)sizeof buf - 1;
    httpd_resp_set_type(r, "application/json");
    return httpd_resp_send(r, buf, n);
}

/* ---------- ids in, ids out ----------
 * Body is {"ids":[...]}; parsed by hand because pulling in a JSON library for
 * one array of integers is not worth the flash. */

static esp_err_t h_tokens(httpd_req_t *r)
{
    static char body[6144];
    static int32_t ids[NE_MAX_ENC];
    static int32_t out[NE_MAX_GEN];

    if (r->content_len > (int)sizeof body - 1) {
        httpd_resp_set_status(r, "413 Payload Too Large");
        return httpd_resp_sendstr(r, "{\"error\":\"body too large\"}");
    }
    int total = 0;
    while (total < r->content_len && total < (int)sizeof body - 1) {
        int got = httpd_req_recv(r, body + total, sizeof body - 1 - total);
        if (got <= 0) return ESP_FAIL;
        total += got;
    }
    body[total] = 0;

    const char *p = strstr(body, "\"ids\"");
    if (p) p = strchr(p, '[');
    if (!p) {
        httpd_resp_set_status(r, "400 Bad Request");
        return httpd_resp_sendstr(r, "{\"error\":\"expected {\\\"ids\\\":[...]}\"}");
    }
    p++;
    int n = 0;
    while (*p && *p != ']' && n < NE_MAX_ENC) {
        while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        if (*p == ']' || !*p) break;
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (v < 0 || v >= NE_VOCAB) {
            httpd_resp_set_status(r, "400 Bad Request");
            return httpd_resp_sendstr(r, "{\"error\":\"id out of range\"}");
        }
        ids[n++] = (int32_t)v;
        p = end;
    }
    if (n == 0) {
        httpd_resp_set_status(r, "400 Bad Request");
        return httpd_resp_sendstr(r, "{\"error\":\"no ids\"}");
    }

    if (net_selftest_failed()) {
        httpd_resp_set_status(r, "500 Internal Server Error");
        return httpd_resp_sendstr(r,
            "{\"error\":\"boot self-tests failed; this board's answers are not trustworthy\"}");
    }
    if (!net_infer_try_lock()) {
        httpd_resp_set_status(r, "503 Service Unavailable");
        httpd_resp_set_hdr(r, "Retry-After", "30");
        return httpd_resp_sendstr(r, "{\"error\":\"busy running another query\"}");
    }
    float enc_ms = 0, tok_ms = 0;
    int n_out = g_infer(ids, n, out, &enc_ms, &tok_ms);
    net_infer_unlock();
    if (n_out < 0) {
        httpd_resp_set_status(r, "500 Internal Server Error");
        return httpd_resp_sendstr(r, "{\"error\":\"inference failed\"}");
    }

    static char resp[1024];
    int w = snprintf(resp, sizeof resp, "{\"gen_ids\":[");
    for (int i = 0; i < n_out && w < (int)sizeof resp - 24; i++)
        w += snprintf(resp + w, sizeof resp - w, "%s%d", i ? "," : "", (int)out[i]);
    w += snprintf(resp + w, sizeof resp - w,
                  "],\"enc_ms\":%.0f,\"tok_ms\":%.1f,\"mode\":\"onboard\",\"path\":\"%c\"}",
                  (double)enc_ms, (double)tok_ms,
#ifdef NE_WARMSCHEMA
                  g_infer_path
#else
                  'e'
#endif
                  );
    /* snprintf returns what it WOULD have written; sending that length would
     * read past the buffer. */
    if (w > (int)sizeof resp - 1) w = (int)sizeof resp - 1;
    httpd_resp_set_type(r, "application/json");
    return httpd_resp_send(r, resp, w);
}

static void start_http(void)
{
    httpd_handle_t srv = NULL;
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    /* ne_step alone holds ~10 KB of float locals and ne_encode adds more. A
     * server task smaller than the main task overflows its stack inside the
     * handler: no panic, the connection just dies. */
    cfg.stack_size = 32768;
    cfg.lru_purge_enable = true;
    cfg.uri_match_fn = httpd_uri_match_wildcard;
    /* an encode can hold the worker for tens of seconds */
    cfg.recv_wait_timeout = 120;
    cfg.send_wait_timeout = 120;
    /* pin the server to core 0: core 1 runs the GEMV worker with its scheduler
     * permanently suspended and will never service a socket */
    cfg.core_id = 0;

    if (httpd_start(&srv, &cfg) != ESP_OK) { ESP_LOGE(TAG, "httpd failed"); return; }

    static const httpd_uri_t routes[] = {
#if CONFIG_NEEDLE_WEB_UI
        { .uri = "/",          .method = HTTP_GET,  .handler = h_index  },
        { .uri = "/bpe.js",    .method = HTTP_GET,  .handler = h_bpe    },
        { .uri = "/tool_retrieval.js", .method = HTTP_GET, .handler = h_retr },
        { .uri = "/needle.js", .method = HTTP_GET,  .handler = h_needle },
        { .uri = "/vocab.txt", .method = HTTP_GET,  .handler = h_vocab  },
#endif
        { .uri = "/api/status",.method = HTTP_GET,  .handler = h_status },
        { .uri = "/api/tokens",.method = HTTP_POST, .handler = h_tokens },
    };
    for (size_t i = 0; i < sizeof routes / sizeof *routes; i++)
        httpd_register_uri_handler(srv, &routes[i]);
    ESP_LOGI(TAG, "http up");
}

static void on_got_ip(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    static bool started;                 /* GOT_IP fires again on DHCP renewal */
    ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
    snprintf(g_ip, sizeof g_ip, IPSTR, IP2STR(&e->ip_info.ip));
    ESP_LOGI(TAG, "http://%s/", g_ip);
    if (!started) { started = true; start_http(); }
}

static void on_eth(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (id == ETHERNET_EVENT_CONNECTED)      ESP_LOGI(TAG, "link up");
    else if (id == ETHERNET_EVENT_DISCONNECTED) ESP_LOGW(TAG, "link down");
}

void net_start(ne_infer_fn infer)
{
    g_infer = infer;

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_config_t nc = ESP_NETIF_DEFAULT_ETH();
    esp_netif_t *netif = esp_netif_new(&nc);

    eth_mac_config_t mac_cfg = ETH_MAC_DEFAULT_CONFIG();
    eth_esp32_emac_config_t esp_cfg = ETH_ESP32_EMAC_DEFAULT_CONFIG();
    esp_eth_mac_t *mac = esp_eth_mac_new_esp32(&esp_cfg, &mac_cfg);

    eth_phy_config_t phy_cfg = ETH_PHY_DEFAULT_CONFIG();
    phy_cfg.phy_addr = PHY_ADDR;
    phy_cfg.reset_gpio_num = PHY_RESET_IO;
    esp_eth_phy_t *phy = esp_eth_phy_new_ip101(&phy_cfg);

    esp_eth_config_t eth_cfg = ETH_DEFAULT_CONFIG(mac, phy);
    esp_eth_handle_t eth = NULL;
    ESP_ERROR_CHECK(esp_eth_driver_install(&eth_cfg, &eth));
    ESP_ERROR_CHECK(esp_netif_attach(netif, esp_eth_new_netif_glue(eth)));
    ESP_ERROR_CHECK(esp_event_handler_register(ETH_EVENT, ESP_EVENT_ANY_ID, on_eth, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_ETH_GOT_IP, on_got_ip, NULL));
    ESP_ERROR_CHECK(esp_eth_start(eth));
    ESP_LOGI(TAG, "ethernet started, waiting for DHCP");
}

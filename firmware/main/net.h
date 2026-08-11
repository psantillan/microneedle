/* Ethernet + HTTP front end for the board-hosted demo. */
#ifndef NET_H
#define NET_H
#include <stdint.h>

/* Run one query. Returns the number of generated tokens, or -1.
 * Implemented in fw_main.c, which owns the engine context. */
typedef int (*ne_infer_fn)(const int32_t *ids, int n, int32_t *out,
                           float *enc_ms, float *tok_ms);

void net_start(ne_infer_fn infer);
const char *net_ip(void);

/* JSON array of the boot self-test lines, for /api/status. */
const char *net_selftest_json(void);

/* Non-zero if any boot check failed. A board whose kernels are wrong will
 * still happily answer queries, so this is surfaced rather than assumed. */
int net_selftest_failed(void);

/* The engine holds one model in file-scope state and both the HTTP handler and
 * the serial console can start a query. Both take this. Returns 0 if another
 * query already holds it. */
int  net_infer_try_lock(void);
void net_infer_unlock(void);
#endif

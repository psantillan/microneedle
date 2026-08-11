/* Host ask runner: one prompt in, generated ids out.
 *
 * demos/ask.py drives this for the no-hardware path: it tokenizes in Python,
 * hands the encoder ids over (argv or stdin), and decodes what comes back.
 * The engine is the same scalar build the parity harness certifies, so the
 * answer is the reference answer, not an approximation of the board's.
 *
 *   needle_ask model.npk 1060 3076 ...       ids as arguments
 *   echo "1060 3076 ..." | needle_ask model.npk
 *
 * stdout: one line of generated token ids, space-separated.
 * stderr: one timing line (prefill to first token, then ms/token).
 */
#include "needle_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint8_t *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc(*len);
    if (!p) { perror("malloc"); exit(1); }
    if (fread(p, 1, *len, f) != *len) { perror("fread"); exit(1); }
    fclose(f);
    return p;
}

/* First callback marks the end of prefill (plus one decode step, which is
 * close enough for a demo timing line). */
static clock_t t_first;
static void first_tok(void *user, int32_t token)
{
    (void)user; (void)token;
    if (!t_first) t_first = clock();
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.npk [ids...]   (or ids on stdin)\n", argv[0]);
        return 2;
    }
    size_t len;
    uint8_t *npk = slurp(argv[1], &len);

    ne_ctx_t ctx;
    memset(&ctx, 0, sizeof ctx);
    if (ne_load(&ctx, npk, len) != 0) { fprintf(stderr, "bad npk\n"); return 1; }
    if (ne_alloc(&ctx) != 0) { fprintf(stderr, "alloc failed\n"); return 1; }

    int32_t enc[NE_MAX_ENC];
    int n = 0;
    if (argc > 2) {
        for (int i = 2; i < argc && n < NE_MAX_ENC; i++)
            enc[n++] = (int32_t)strtol(argv[i], NULL, 10);
    } else {
        long v;
        while (n < NE_MAX_ENC && scanf("%ld", &v) == 1) enc[n++] = (int32_t)v;
    }
    if (n == 0) { fprintf(stderr, "no encoder ids\n"); return 2; }

    int32_t out[NE_MAX_GEN];
    clock_t t0 = clock();
    int n_out = ne_generate_cb(&ctx, enc, n, out, NE_MAX_GEN - 1, first_tok, NULL);
    clock_t t1 = clock();
    if (n_out < 0) { fprintf(stderr, "encode failed (too many ids?)\n"); return 1; }

    for (int i = 0; i < n_out; i++)
        printf("%d%c", out[i], i + 1 == n_out ? '\n' : ' ');
    if (n_out == 0) printf("\n");

    double ms = (double)(t1 - t0) * 1000.0 / CLOCKS_PER_SEC;
    double enc_ms = (double)((t_first ? t_first : t1) - t0) * 1000.0 / CLOCKS_PER_SEC;
    fprintf(stderr, "T enc_ms=%.0f tok_ms=%.1f total_ms=%.0f n=%d\n",
            enc_ms, n_out ? (ms - enc_ms) / n_out : 0.0, ms, n_out);

    ne_free(&ctx);
    free(npk);
    return 0;
}

/* Host parity harness: run the C engine over the fixture vectors and diff
 * against the JAX oracle (the oracle is dumped with group-32 fake
 * quantization; the shipped artifact is group-512, so the oracle is the
 * slightly stricter reference and a disagreement still points at the engine).
 *
 * Vector file format (from ref_to_vectors.py), one case per line:
 *   id n_enc <enc ids...> n_ref <ref ids...>
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

#ifdef NE_PIE
/* Host stand-ins for the firmware's dual-core splitters: run the two halves
 * SEQUENTIALLY, worker half first. If the maths is truly range-partitioned --
 * disjoint outputs, no order dependence -- the result cannot change; if
 * someone introduces a dependence between the halves, this host gate catches
 * it before the board turns it into a race. */
static void host_split_rows(const ne_gemv_job_t *j, uint32_t rows)
{ ne_gemv_rows(j, rows / 2, rows); ne_gemv_rows(j, 0, rows / 2); }
static void host_split_gemm(const ne_gemm_job_t *j, uint32_t rows)
{ ne_gemm_rows(j, rows / 2, rows); ne_gemm_rows(j, 0, rows / 2); }
static void host_split_attn(const ne_attn_job_t *j, uint32_t h)
{ ne_attn_heads(j, h / 2, h); ne_attn_heads(j, 0, h / 2); }
static void host_split_for(void (*fn)(void *, int, int), void *arg, int n)
{ fn(arg, n / 2, n); fn(arg, 0, n / 2); }
#endif

int main(int argc, char **argv)
{
#ifdef NE_PIE
    ne_split_rows = host_split_rows;
    ne_split_gemm = host_split_gemm;
    ne_split_attn = host_split_attn;
    ne_split_for  = host_split_for;
#endif
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.npk vectors.txt\n", argv[0]);
        return 2;
    }
    size_t len;
    uint8_t *npk = slurp(argv[1], &len);

    ne_ctx_t ctx;
    memset(&ctx, 0, sizeof ctx);
    if (ne_load(&ctx, npk, len) != 0) { fprintf(stderr, "bad npk\n"); return 1; }
    if (ne_alloc(&ctx) != 0) { fprintf(stderr, "alloc failed\n"); return 1; }

    FILE *vf = fopen(argv[2], "r");
    if (!vf) { perror(argv[2]); return 1; }

    char id[64];
    int n_enc, n_ref, pass = 0, total = 0;
    while (fscanf(vf, "%63s %d", id, &n_enc) == 2) {
        int32_t enc[NE_MAX_ENC], ref[NE_MAX_GEN], got[NE_MAX_GEN];
        int bad = (n_enc < 1 || n_enc > NE_MAX_ENC);
        for (int i = 0; !bad && i < n_enc; i++) bad |= fscanf(vf, "%d", &enc[i]) != 1;
        bad |= fscanf(vf, "%d", &n_ref) != 1 || n_ref < 0 || n_ref > NE_MAX_GEN;
        for (int i = 0; !bad && i < n_ref; i++) bad |= fscanf(vf, "%d", &ref[i]) != 1;
        if (bad) { fprintf(stderr, "malformed vector file at case '%s'\n", id); return 2; }

        clock_t t0 = clock();
        int n_got = ne_generate(&ctx, enc, n_enc, got, NE_MAX_GEN - 1);
        if (n_got < 0) { fprintf(stderr, "encode failed for '%s'\n", id); return 1; }
        double ms = 1000.0 * (double)(clock() - t0) / CLOCKS_PER_SEC;

        int same = (n_got == n_ref) && !memcmp(got, ref, n_ref * sizeof(int32_t));
        total++;
        pass += same;
        printf("%-22s %s  enc=%-4d ref=%-3d got=%-3d  %6.0f ms\n",
               id, same ? "PASS" : "FAIL", n_enc, n_ref, n_got, ms);
        if (!same) {
            printf("  ref:");
            for (int i = 0; i < n_ref; i++) printf(" %d", ref[i]);
            printf("\n  got:");
            for (int i = 0; i < n_got; i++) printf(" %d", got[i]);
            printf("\n  first divergence at index ");
            int i = 0;
            while (i < n_ref && i < n_got && ref[i] == got[i]) i++;
            printf("%d (ref=%d got=%d)\n", i,
                   i < n_ref ? ref[i] : -1, i < n_got ? got[i] : -1);
        }
    }
    printf("\n%d/%d cases match the JAX oracle token-for-token\n", pass, total);
    ne_free(&ctx);
    return pass == total ? 0 : 1;
}

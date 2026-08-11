/* Time 100 reps of the eval_tools vector set with/without grammar logits skip. */
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
    if (!p || fread(p, 1, *len, f) != *len) { perror("slurp"); exit(1); }
    fclose(f);
    return p;
}

typedef struct {
    int n_enc;
    int32_t enc[NE_MAX_ENC];
} case_t;

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s model.npk vectors.txt reps\n", argv[0]);
        return 2;
    }
    int reps = atoi(argv[3]);
    if (reps < 1) reps = 1;

    size_t len;
    uint8_t *npk = slurp(argv[1], &len);
    ne_ctx_t ctx;
    memset(&ctx, 0, sizeof ctx);
    if (ne_load(&ctx, npk, len) != 0 || ne_alloc(&ctx) != 0) {
        fprintf(stderr, "load/alloc failed\n");
        return 1;
    }

    case_t cases[32];
    int n_cases = 0;
    FILE *vf = fopen(argv[2], "r");
    if (!vf) { perror(argv[2]); return 1; }
    char id[64];
    int n_enc, n_ref, bad;
    while (n_cases < 32 && fscanf(vf, "%63s %d", id, &n_enc) == 2) {
        case_t *c = &cases[n_cases];
        c->n_enc = n_enc;
        bad = 0;
        for (int i = 0; i < n_enc; i++) bad |= fscanf(vf, "%d", &c->enc[i]) != 1;
        bad |= fscanf(vf, "%d", &n_ref) != 1;
        for (int i = 0; i < n_ref; i++) { int dummy; bad |= fscanf(vf, "%d", &dummy) != 1; }
        if (bad) { fprintf(stderr, "bad vector file\n"); return 2; }
        n_cases++;
    }
    fclose(vf);

    int32_t out[NE_MAX_GEN];
    for (int mode = 0; mode < 2; mode++) {
        ne_grammar_force = mode;   /* 0 = off, 1 = on */
        ne_grammar_skips = 0;
        long long tokens = 0;
        clock_t t0 = clock();
        for (int r = 0; r < reps; r++) {
            for (int c = 0; c < n_cases; c++) {
                int n = ne_generate(&ctx, cases[c].enc, cases[c].n_enc,
                                    out, NE_MAX_GEN - 1);
                if (n < 0) { fprintf(stderr, "encode fail\n"); return 1; }
                tokens += n;
            }
        }
        double sec = (double)(clock() - t0) / (double)CLOCKS_PER_SEC;
        printf("grammar_force=%d  cases=%d  reps=%d  tokens=%lld  skips=%d  cpu_s=%.3f\n",
               mode, n_cases, reps, tokens, ne_grammar_skips, sec);
    }
    ne_free(&ctx);
    free(npk);
    return 0;
}

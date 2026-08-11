/* Product warm-schema F=4 host runner.
 *
 *   needle_warmschema model.npk vectors.txt [--mutate|--wrong-schema]
 *
 * Default: capture once from the first case (schema key), then for every case
 * print exact vs warm token ids and row-layer stats. Cache bytes and capture
 * wall time go to stderr.
 *
 * --mutate: after capture, corrupt one cached K element, show warm diverges,
 *           restore, show warm matches exact again (on the first case only).
 * --wrong-schema: capture from first case, attempt warm on a second case that
 *           must carry a different schema (or a synthetic key flip); expect
 *           ne_generate_warm to return -1 (guard fires, no silent use).
 *
 * Scalar fp32 path only (fixture-certified reference).
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

static void print_ids(const char *id, const char *tag, const int32_t *got, int n)
{
    printf("%s %s:", id, tag);
    for (int i = 0; i < n; i++) printf(" %d", got[i]);
    printf("\n");
    fflush(stdout);
}

static int same_ids(const int32_t *a, int na, const int32_t *b, int nb)
{
    if (na != nb) return 0;
    return memcmp(a, b, (size_t)na * sizeof(int32_t)) == 0;
}

typedef struct {
    char id[64];
    int n_enc;
    int32_t enc[NE_MAX_ENC];
    int n_ref;
    int32_t ref[NE_MAX_GEN];
} case_t;

static int load_cases(const char *path, case_t **out, int *nout)
{
    FILE *vf = fopen(path, "r");
    if (!vf) { perror(path); return -1; }
    int cap = 8, n = 0;
    case_t *cs = malloc((size_t)cap * sizeof(case_t));
    if (!cs) { fclose(vf); return -1; }
    while (1) {
        case_t c;
        memset(&c, 0, sizeof c);
        if (fscanf(vf, "%63s %d", c.id, &c.n_enc) != 2) break;
        int bad = (c.n_enc < 1 || c.n_enc > NE_MAX_ENC);
        for (int i = 0; !bad && i < c.n_enc; i++)
            bad |= fscanf(vf, "%d", &c.enc[i]) != 1;
        bad |= fscanf(vf, "%d", &c.n_ref) != 1 || c.n_ref < 0 || c.n_ref > NE_MAX_GEN;
        for (int i = 0; !bad && i < c.n_ref; i++)
            bad |= fscanf(vf, "%d", &c.ref[i]) != 1;
        if (bad) {
            fprintf(stderr, "malformed vector file at case '%s'\n", c.id);
            free(cs); fclose(vf); return -1;
        }
        if (n == cap) {
            cap *= 2;
            case_t *ncs = realloc(cs, (size_t)cap * sizeof(case_t));
            if (!ncs) { free(cs); fclose(vf); return -1; }
            cs = ncs;
        }
        cs[n++] = c;
    }
    fclose(vf);
    *out = cs;
    *nout = n;
    return 0;
}

/* Flip one schema token so the key no longer matches (token stays in-vocab). */
static void flip_schema_key(int32_t *enc, int n)
{
    for (int i = 0; i < n; i++) {
        if (enc[i] == 5) { /* <tools> */
            if (i + 1 < n) {
                enc[i + 1] ^= 1;
                if (enc[i + 1] < 0 || enc[i + 1] >= NE_VOCAB) enc[i + 1] = 6;
            }
            return;
        }
    }
}

int main(int argc, char **argv)
{
    int do_mutate = 0, do_wrong = 0;
    const char *npk_path = NULL, *vec_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mutate")) do_mutate = 1;
        else if (!strcmp(argv[i], "--wrong-schema")) do_wrong = 1;
        else if (!npk_path) npk_path = argv[i];
        else if (!vec_path) vec_path = argv[i];
        else {
            fprintf(stderr, "usage: %s model.npk vectors.txt [--mutate|--wrong-schema]\n",
                    argv[0]);
            return 2;
        }
    }
    if (!npk_path || !vec_path) {
        fprintf(stderr, "usage: %s model.npk vectors.txt [--mutate|--wrong-schema]\n",
                argv[0]);
        return 2;
    }
    if (do_mutate && do_wrong) {
        fprintf(stderr, "pick at most one of --mutate / --wrong-schema\n");
        return 2;
    }

    size_t len;
    uint8_t *npk = slurp(npk_path, &len);

    ne_ctx_t ctx;
    memset(&ctx, 0, sizeof ctx);
    if (ne_load(&ctx, npk, len) != 0) { fprintf(stderr, "bad npk\n"); return 1; }
    if (ne_alloc(&ctx) != 0) { fprintf(stderr, "alloc failed\n"); return 1; }

    case_t *cases = NULL;
    int ncases = 0;
    if (load_cases(vec_path, &cases, &ncases) != 0 || ncases < 1) {
        fprintf(stderr, "no cases in %s\n", vec_path);
        return 1;
    }

    fprintf(stderr, "ne_warm_F=%d  cases=%d\n", NE_WARM_F, ncases);

    /* ---- Capture once from first case ---- */
    clock_t t0 = clock();
    ne_schema_cache_t *cache = ne_schema_capture(&ctx, cases[0].enc, cases[0].n_enc);
    clock_t t1 = clock();
    if (!cache) {
        fprintf(stderr, "schema capture failed on '%s'\n", cases[0].id);
        return 1;
    }
    double cap_ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
    size_t cbytes = ne_schema_cache_bytes(cache);
    int cns = ne_schema_cache_ns(cache);
    fprintf(stderr, "cache: ns=%d bytes=%zu (%.2f KiB) capture_ms=%.2f ref=%s\n",
            cns, cbytes, cbytes / 1024.0, cap_ms, cases[0].id);
    printf("CACHE ns=%d bytes=%zu capture_ms=%.2f\n", cns, cbytes, cap_ms);

    if (do_mutate) {
        /* Mutation that can fail: zero one cached K row, prove the warm
         * encoder state diverges (enc_x max-abs), restore, prove it matches
         * again. Token identity is NOT required to flip — at F=4 phase B
         * often recovers the same greedy tokens — but enc_x must move.
         * Also zero residual for a token-level divergence record. */
        case_t *c = &cases[0];
        for (int i = 1; i < ncases; i++) {
            if (ne_schema_cache_matches(cache, cases[i].enc, cases[i].n_enc)) {
                c = &cases[i];
                break;
            }
        }
        int32_t got_e[NE_MAX_GEN], got_w[NE_MAX_GEN], got_r[NE_MAX_GEN];
        int n_e = ne_generate(&ctx, c->enc, c->n_enc, got_e, NE_MAX_GEN - 1);
        if (n_e < 0) { fprintf(stderr, "exact failed\n"); return 1; }

        /* Clean warm encode → snapshot enc_x. */
        if (ne_encode_warm(&ctx, c->enc, c->n_enc, cache) != 0) {
            fprintf(stderr, "clean warm encode failed\n"); return 1;
        }
        int S = ctx.enc_len;
        size_t xbytes = (size_t)S * NE_DMODEL * sizeof(float);
        float *clean = malloc(xbytes);
        if (!clean) { perror("malloc"); return 1; }
        memcpy(clean, ctx.enc_x, xbytes);

        int mrow = cns / 2;
        float *krow = ne_schema_cache_k_row(cache, NE_WARM_F - 1, mrow);
        if (!krow) { fprintf(stderr, "k_row unavailable\n"); free(clean); return 1; }
        float saved_k[NE_KVDIM];
        memcpy(saved_k, krow, sizeof saved_k);
        for (int d = 0; d < NE_KVDIM; d++) krow[d] = 0.f;

        if (ne_encode_warm(&ctx, c->enc, c->n_enc, cache) != 0) {
            fprintf(stderr, "mutated warm encode failed\n"); free(clean); return 1;
        }
        float max_abs = 0.f;
        for (size_t i = 0; i < (size_t)S * NE_DMODEL; i++) {
            float d = ctx.enc_x[i] - clean[i];
            if (d < 0) d = -d;
            if (d > max_abs) max_abs = d;
        }
        int state_div = max_abs > 1e-5f;
        printf("%s k_mutate: layer=%d row=%d enc_x_max_abs=%.6g state_diverged=%d\n",
               c->id, NE_WARM_F - 1, mrow, max_abs, state_div);

        memcpy(krow, saved_k, sizeof saved_k);  /* restore K */
        if (ne_encode_warm(&ctx, c->enc, c->n_enc, cache) != 0) {
            fprintf(stderr, "restored warm encode failed\n"); free(clean); return 1;
        }
        float max_rest = 0.f;
        for (size_t i = 0; i < (size_t)S * NE_DMODEL; i++) {
            float d = ctx.enc_x[i] - clean[i];
            if (d < 0) d = -d;
            if (d > max_rest) max_rest = d;
        }
        int state_restored = max_rest <= 1e-5f;
        printf("%s k_restore: enc_x_max_abs=%.6g state_restored=%d\n",
               c->id, max_rest, state_restored);

        /* Token-level record: residual zero-all (phase-B splice) must flip tokens. */
        float *xsave = malloc((size_t)cns * NE_DMODEL * sizeof(float));
        if (!xsave) { perror("malloc"); free(clean); return 1; }
        for (int r = 0; r < cns; r++) {
            float *xr = ne_schema_cache_x_row(cache, r);
            memcpy(xsave + (size_t)r * NE_DMODEL, xr, NE_DMODEL * sizeof(float));
            for (int d = 0; d < NE_DMODEL; d++) xr[d] = 0.f;
        }
        int n_w = ne_generate_warm(&ctx, c->enc, c->n_enc, got_w, NE_MAX_GEN - 1, cache);
        int tok_div = (n_w >= 0) && !same_ids(got_e, n_e, got_w, n_w);
        print_ids(c->id, "exact", got_e, n_e);
        print_ids(c->id, "warm_xmut", got_w, n_w < 0 ? 0 : n_w);
        printf("%s x_mutate_all: token_diverged=%d\n", c->id, tok_div);

        for (int r = 0; r < cns; r++) {
            float *xr = ne_schema_cache_x_row(cache, r);
            memcpy(xr, xsave + (size_t)r * NE_DMODEL, NE_DMODEL * sizeof(float));
        }
        int n_r = ne_generate_warm(&ctx, c->enc, c->n_enc, got_r, NE_MAX_GEN - 1, cache);
        int tok_rest = (n_r >= 0) && same_ids(got_e, n_e, got_r, n_r);
        print_ids(c->id, "warm_xrestored", got_r, n_r < 0 ? 0 : n_r);
        printf("%s x_restore: token_identical=%d\n", c->id, tok_rest);

        int ok = state_div && state_restored && tok_div && tok_rest;
        printf("MUTATE_OK %d\n", ok ? 1 : 0);
        free(xsave); free(clean);
        ne_schema_cache_free(cache);
        free(cases); ne_free(&ctx); free(npk);
        return ok ? 0 : 1;
    }

    if (do_wrong) {
        /* Build a prompt with a deliberately wrong schema key. Prefer a second
         * case whose schema differs; else flip a token on a copy of case 0. */
        int32_t bad_enc[NE_MAX_ENC];
        int bad_n = cases[0].n_enc;
        memcpy(bad_enc, cases[0].enc, (size_t)bad_n * sizeof(int32_t));
        int used_other = 0;
        for (int i = 1; i < ncases; i++) {
            if (!ne_schema_cache_matches(cache, cases[i].enc, cases[i].n_enc)) {
                bad_n = cases[i].n_enc;
                memcpy(bad_enc, cases[i].enc, (size_t)bad_n * sizeof(int32_t));
                used_other = i;
                break;
            }
        }
        if (used_other == 0)
            flip_schema_key(bad_enc, bad_n);

        int match = ne_schema_cache_matches(cache, bad_enc, bad_n);
        int32_t got_w[NE_MAX_GEN];
        int n_w = ne_generate_warm(&ctx, bad_enc, bad_n, got_w, NE_MAX_GEN - 1, cache);
        printf("WRONG_SCHEMA matches=%d warm_rc=%d (expect matches=0 warm_rc=-1)\n",
               match, n_w);
        fprintf(stderr, "wrong-schema guard: matches=%d warm_rc=%d used_case=%s\n",
                match, n_w,
                used_other ? cases[used_other].id : "(flipped-key case0)");
        int ok = (match == 0 && n_w < 0);
        printf("WRONG_SCHEMA_OK %d\n", ok ? 1 : 0);
        ne_schema_cache_free(cache);
        free(cases); ne_free(&ctx); free(npk);
        return ok ? 0 : 1;
    }

    /* ---- Exact vs warm for all cases ---- */
    int n_ok = 0, n_skip = 0;
    for (int i = 0; i < ncases; i++) {
        case_t *c = &cases[i];
        int32_t got_e[NE_MAX_GEN], got_w[NE_MAX_GEN];

        int n_e = ne_generate(&ctx, c->enc, c->n_enc, got_e, NE_MAX_GEN - 1);
        if (n_e < 0) {
            fprintf(stderr, "exact encode failed for '%s'\n", c->id);
            return 1;
        }
        print_ids(c->id, "exact", got_e, n_e);

        if (!ne_schema_cache_matches(cache, c->enc, c->n_enc)) {
            printf("%s warm: SKIP key_mismatch\n", c->id);
            printf("%s rows: 0 0 0 0 0 %d\n", c->id, NE_WARM_F);
            n_skip++;
            continue;
        }

        int n_w = ne_generate_warm(&ctx, c->enc, c->n_enc, got_w, NE_MAX_GEN - 1, cache);
        if (n_w < 0) {
            fprintf(stderr, "warm encode failed for '%s' (unexpected after match)\n",
                    c->id);
            return 1;
        }
        print_ids(c->id, "warm", got_w, n_w);
        printf("%s rows: %d %d %d %d %d %d\n", c->id,
               ne_warm_nq, ne_warm_ns, ne_warm_ntot,
               ne_warm_row_layers, ne_warm_row_layers_full, NE_WARM_F);
        if (same_ids(got_e, n_e, got_w, n_w)) n_ok++;
        fflush(stdout);
    }
    printf("SUMMARY identical=%d total=%d skipped_key=%d\n",
           n_ok, ncases - n_skip, n_skip);

    ne_schema_cache_free(cache);
    free(cases);
    ne_free(&ctx);
    free(npk);
    return 0;
}

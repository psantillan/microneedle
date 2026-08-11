/* Latent-path trace runner: the host engine with NE_TRACE hooks, one binary
 * trace file per case. Same vector-file format as main_host.c, but the
 * reference ids are ignored -- this tool records what the model does inside,
 * it does not judge the outside.
 *
 *   needle_trace model.npk vectors.txt outdir
 *
 * Writes outdir/model.bin (dequantized embedding + decoder final norm, once)
 * and outdir/<id>.bin per case, and prints "<id> got: <ids...>" so the driver
 * can decode the generations without parsing the binary. Record format is in
 * needle_engine.h next to ne_trace_f. Scalar build only: the fp32 path is the
 * fixture-certified reference, so the story it tells is the board's story. */
#include "needle_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s model.npk vectors.txt outdir\n", argv[0]);
        return 2;
    }
    size_t len;
    uint8_t *npk = slurp(argv[1], &len);

    ne_ctx_t ctx;
    memset(&ctx, 0, sizeof ctx);
    if (ne_load(&ctx, npk, len) != 0) { fprintf(stderr, "bad npk\n"); return 1; }
    if (ne_alloc(&ctx) != 0) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* NE_ABLATE="enc:3:5" | "dself:L:h" | "dcross:L:h" zeroes one head's
     * attention output for the whole run; outdir "-" skips the binary dumps
     * so an ablation sweep is generations-only. */
    const char *ab = getenv("NE_ABLATE");
    if (ab) {
        char kind[16];
        if (sscanf(ab, "%15[a-z]:%d:%d", kind, &ne_ablate_layer, &ne_ablate_head) == 3)
            ne_ablate_kind = !strcmp(kind, "enc") ? 1 : !strcmp(kind, "dself") ? 2
                           : !strcmp(kind, "dcross") ? 3 : 0;
        if (!ne_ablate_kind) { fprintf(stderr, "bad NE_ABLATE '%s'\n", ab); return 2; }
    }
    const int dumps = strcmp(argv[3], "-") != 0;

    char path[512];
    if (dumps) {
        snprintf(path, sizeof path, "%s/model.bin", argv[3]);
        ne_trace_f = fopen(path, "wb");
        if (!ne_trace_f) { perror(path); return 1; }
        ne_trace_dump_model(&ctx);
        fclose(ne_trace_f);
        ne_trace_f = NULL;
        if (getenv("NE_DUMP_WEIGHTS")) {
            snprintf(path, sizeof path, "%s/weights.bin", argv[3]);
            FILE *wf = fopen(path, "wb");
            if (!wf) { perror(path); return 1; }
            ne_trace_dump_weights(&ctx, wf);
            fclose(wf);
        }
    }

    FILE *vf = fopen(argv[2], "r");
    if (!vf) { perror(argv[2]); return 1; }

    char id[64];
    int n_enc, n_ref;
    while (fscanf(vf, "%63s %d", id, &n_enc) == 2) {
        int32_t enc[NE_MAX_ENC], ref[NE_MAX_GEN], got[NE_MAX_GEN];
        int bad = (n_enc < 1 || n_enc > NE_MAX_ENC);
        for (int i = 0; !bad && i < n_enc; i++) bad |= fscanf(vf, "%d", &enc[i]) != 1;
        bad |= fscanf(vf, "%d", &n_ref) != 1 || n_ref < 0 || n_ref > NE_MAX_GEN;
        for (int i = 0; !bad && i < n_ref; i++) bad |= fscanf(vf, "%d", &ref[i]) != 1;
        if (bad) { fprintf(stderr, "malformed vector file at case '%s'\n", id); return 2; }

        if (dumps) {
            snprintf(path, sizeof path, "%s/%s.bin", argv[3], id);
            ne_trace_f = fopen(path, "wb");
            if (!ne_trace_f) { perror(path); return 1; }
        }
        int n_got = ne_generate(&ctx, enc, n_enc, got, NE_MAX_GEN - 1);
        if (ne_trace_f) { fclose(ne_trace_f); ne_trace_f = NULL; }

        printf("%s got:", id);
        for (int i = 0; i < n_got; i++) printf(" %d", got[i]);
        printf("\n");
        fflush(stdout);
    }
    fclose(vf);
    ne_free(&ctx);
    free(npk);
    return 0;
}

# Host parity harness. The engine also compiles into the ESP32-P4 firmware
# (firmware/main/CMakeLists.txt points here); this Makefile is host-only.
CFLAGS ?= -O2 -std=c99 -Wall -Wextra
PIE_STUB = ../tools/pie_stub.c

needle_host: needle_engine.c main_host.c needle_engine.h
	$(CC) $(CFLAGS) -o $@ needle_engine.c main_host.c -lm

# compile-only check that the NE_PIE code paths stay warning-free; the .S
# needs the xesppie toolchain, so host cannot link this configuration
pie-check: needle_engine.c needle_engine.h
	$(CC) $(CFLAGS) -DNE_PIE -c needle_engine.c -o /dev/null
	$(CC) $(CFLAGS) -DNE_PIE -DNE_ATTN_I8 -c needle_engine.c -o /dev/null

# latent-path tracer (tools/probe_latent_audio.py drives it); scalar fp32
# path only, which is the fixture-certified reference
needle_trace: needle_engine.c main_trace.c needle_engine.h
	$(CC) $(CFLAGS) -DNE_TRACE -o $@ needle_engine.c main_trace.c -lm

# interactive host runner (demos/ask.py drives it): ids in, generated ids out
needle_ask: needle_engine.c main_ask.c needle_engine.h
	$(CC) $(CFLAGS) -o $@ needle_engine.c main_ask.c -lm

# Host stub-parity of the NE_PIE path (int16 attention quantization, default).
# Same recipe as verify.sh's "vector path logic" step.
needle_pie: needle_engine.c main_host.c needle_engine.h $(PIE_STUB)
	$(CC) $(CFLAGS) -DNE_PIE -o $@ needle_engine.c main_host.c $(PIE_STUB) -lm

# int8 attention quantization experiment: same stub-parity configuration with
# NE_ATTN_I8 (host-only; does not change the firmware build).
needle_i8: needle_engine.c main_host.c needle_engine.h $(PIE_STUB)
	$(CC) $(CFLAGS) -DNE_PIE -DNE_ATTN_I8 -o $@ needle_engine.c main_host.c $(PIE_STUB) -lm

# Traced int8 build for margin analysis (logits per step via NE_TRACE).
needle_i8_trace: needle_engine.c main_trace.c needle_engine.h $(PIE_STUB)
	$(CC) $(CFLAGS) -DNE_PIE -DNE_ATTN_I8 -DNE_TRACE -o $@ \
		needle_engine.c main_trace.c $(PIE_STUB) -lm

# A/B wall for grammar logits skip (see RESULTS.md)
bench_grammar: needle_engine.c bench_grammar.c needle_engine.h
	$(CC) $(CFLAGS) -o $@ needle_engine.c bench_grammar.c -lm

# product warm-schema F=4 (tools/eval_warmf4.py); host-only, opt-in
needle_warmschema: needle_engine.c main_warmschema.c needle_engine.h
	$(CC) $(CFLAGS) -DNE_WARMSCHEMA -o $@ needle_engine.c main_warmschema.c -lm

# compile-only: NE_WARMSCHEMA must stay warning-free under -DNE_PIE too
pie-check-warm: needle_engine.c needle_engine.h
	$(CC) $(CFLAGS) -DNE_PIE -DNE_WARMSCHEMA -c needle_engine.c -o /dev/null

test: needle_host
	./needle_host ../weights/needle_v3_g512.npk ../weights/vectors_int4p_int8e.txt

clean:
	rm -f needle_host needle_pie needle_i8 needle_trace needle_i8_trace needle_ask \
		bench_grammar needle_warmschema

.PHONY: pie-check pie-check-warm test clean needle_pie needle_i8 needle_i8_trace

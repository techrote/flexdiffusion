# Bootstrap review checklist

- [ ] Trajectory disabled remains the normal path with no manifest I/O.
- [ ] Only `ed_classic` advertises trajectory capture.
- [ ] Unsupported backends reject an enabled trajectory request before receiving trajectory-specific kwargs.
- [ ] Schedule parsing resolves exact/range/percentage syntax deterministically.
- [ ] Callback indices are stored explicitly and displayed steps are one-based.
- [ ] Recorder never retains `x_samples` after the callback returns.
- [ ] Manifest writes are atomic.
- [ ] Interrupt/failure/completion state is explicit.
- [ ] Capture errors are logged and do not destroy an otherwise successful final image.
- [ ] No exact-resume claim exists yet.

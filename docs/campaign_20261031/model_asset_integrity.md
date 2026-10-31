# Local model asset integrity

`verify_model` verifies each declared file once and requires every local file
that could be an inference asset to appear in the manifest. An unlisted
preferred checkpoint, tokenizer/config file, or unknown auxiliary file is an
error, even when every declared hash is correct. Safetensors index references
must resolve to declared safetensor files inside the model root. Tokenizer file
overrides and versioned tokenizer/config references must also remain inside the
root and refer to declared files.

Adapter configurations are rejected: the plain-model runtime has no declared
contract for independently fingerprinting an adapter's external base model.
This also prevents optional PEFT installation from enabling an untracked
base-model redirect.

Only explicitly named root documentation/license files and Hugging Face
download `.metadata`/`.lock` bookkeeping are exempt. Linked directories are not
accepted. The function returns the original manifest unchanged; existing valid
model revisions and manifest digests retain their meanings.

Locally trained epochs may retain their known `optimizer.pt` and
`EPOCH_RECEIPT.json` sidecars. This exception requires the declared local training
content fingerprint, a matching manifest hash in the epoch receipt, and a valid
recorded optimizer digest. These files are excluded from ordinary inference
hashing because `AutoModel` does not load them. `validate_epoch` still verifies
optimizer bytes before training resume; inference validation does not certify
the optimizer state.

The caller must keep files immutable after verification and throughout loading
and use. Existing native session checks provide an additional directory-wide
check; this change does not add a filesystem locking or concurrent-mutation
framework. Integrity validation establishes byte provenance, not scientific
accuracy or model suitability. No existing model file or manifest is rewritten.

Synthetic regressions cover preferred-checkpoint shadowing, unlisted config and
tokenizer assets, escaping/unlisted index and configuration references, malformed
or unbound training receipts, and declared-file tampering. Read-only verification
also accepts the three existing pinned models and both trained epochs unchanged.

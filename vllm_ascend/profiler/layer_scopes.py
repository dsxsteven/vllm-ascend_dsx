"""Explicit layer labels for this diagnostic collection; no tensor changes."""
import re
import threading
from torch.profiler import record_function

def install_layer_scopes(model):
    if getattr(model, "_dsv4_layer_scopes", False):
        return
    model._dsv4_layer_scopes = True
    count = 0
    for name, module in model.named_modules():
        if name and not re.search(r"(?:^|\.)layers\.\d+$", name):
            continue
        label = "dsv4." + (name or "forward")
        state = threading.local()
        def before(mod, args, label=label, state=state):
            scope = record_function(label)
            scope.__enter__()
            if not hasattr(state, "stack"):
                state.stack = []
            state.stack.append(scope)
        def after(mod, args, output, state=state):
            if getattr(state, "stack", None):
                state.stack.pop().__exit__(None, None, None)
        module.register_forward_pre_hook(before)
        module.register_forward_hook(after, always_call=True)
        count += 1
    print(f"DSV4_PROFILE_LAYER_SCOPES count={count}", flush=True)

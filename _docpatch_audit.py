import ast, io
files = ["escreen/motif_tool.py","escreen/genome_tool.py","escreen/miaomiao_tool.py","escreen/serve.py"]
for f in files:
    src = open(f, encoding="utf-8").read()
    tree = ast.parse(src)
    missing, ok_funcs, ok_classes = [], 0, 0
    mod_doc = ast.get_docstring(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            parent = "?"
            body0 = node.body[0] if node.body else None
            is_doc = (isinstance(body0, ast.Expr) and isinstance(body0.value, ast.Constant)
                      and isinstance(body0.value.value, str))
            if not is_doc:
                missing.append((node.lineno, name, type(node).__name__))
            else:
                if isinstance(node, ast.ClassDef):
                    ok_classes += 1
                else:
                    ok_funcs += 1
    print(f"{f}: module_doc={bool(mod_doc)} classes_documented={ok_classes} functions_documented={ok_funcs}")
    for m in missing:
        print("   MISSING", m)

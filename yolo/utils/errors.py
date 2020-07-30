import functools
import sys
import types

try:
    from absl import flags

    flags.DEFINE_boolean('ALLOW_NONPYTHON_ERROR_ORIGINS', True, 'Allow the use of the errors.with_origin function in code')

    ALLOW_NONPYTHON_ERROR_ORIGINS = flags.FLAGS.ALLOW_NONPYTHON_ERROR_ORIGINS
except:
    ALLOW_NONPYTHON_ERROR_ORIGINS = True


if sys.version_info >= (3, 8):
    code_replace = types.CodeType.replace
else:
    # Shim for Python <= 3.7
    def code_replace(co: types.CodeType, kwargs_new: dict):
        kwarg_names = dir(co)

        for kwarg in kwarg_names[:]:
            if not kwarg.startswith('co_'):
                kwarg_names.remove(kwarg)

        kwargs = {k[3:]: getattr(co, k) for k in kwarg_names}
        kwargs.update(kwargs_new)

        return types.CodeType(
            kwargs['argcount'],
            kwargs['kwonlyargcount'],
            kwargs['nlocals'],
            kwargs['stacksize'],
            kwargs['flags'],
            kwargs['code'],
            kwargs['consts'],
            kwargs['names'],
            kwargs['varnames'],
            kwargs['filename'],
            kwargs['name'],
            kwargs['firstlineno'],
            kwargs['lnotab'],
            kwargs['freevars'],
            kwargs['cellvars'],
        )

def with_origin(e, filename, lineno, kwargs_new=None, cause=None):
    """
    Create a fake Python code object to simulate a Python error coming from
    non-Python code.

    Usage:
    ```
    from yolo.utils import errors
    errors.with_origin(SyntaxError('Unknown attribute: policy'), 'yolov3.cfg', 21, {'name': '[net] #1'})()
    ```

    Note:
    If you want to raise a warning, use the warnings package instead:
    ```
    import warnings
    warnings.showwarning('Unknown attribute: policy', SyntaxWarning, 'yolov3.cfg', 21)
    ```
    """
    if ALLOW_NONPYTHON_ERROR_ORIGINS:
        fake_source = "\n" * (lineno - 1) + ("raise e" if cause is None else "raise e from cause")
        code = compile(fake_source, filename, 'exec')
        if kwargs_new is not None:
            code = code_replace(code, kwargs_new)
        context = {'e': e} if cause is None else {'e': e, 'cause': cause}
        return functools.partial(exec, code, context)
    else:
        e_with_info = Exception(f"An issue occurred in {filename}:{lineno}.")
        if cause is None:
            raise e from e_with_info
        else:
            try:
                raise e_with_info from cause
            except Exception as e_with_info:
                raise e from e_with_info

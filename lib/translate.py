import argparse
import json
import numpy as np


class Translate:
    def main(self):
        parser = argparse.ArgumentParser(description="Translate weights to GLSL")
        parser.add_argument('filename')
        parser.add_argument("-s", "--silent",     action="store_true")
        parser.add_argument("-p", "--precision",  type=int, default=4, help="decimal digits for weights (default: 4)")
        parser.add_argument("-c", "--coding",     type=int, default=3)
        parser.add_argument("-e", "--mapping",    choices=["polar", "fourier", "legacy"], default="polar")
        parser.add_argument("-k", "--colorspace", choices=["rgb", "ycbcr", "yuv"], default="ycbcr")
        parser.add_argument("-a", "--activation", choices=["sine", "relu"], default="sine")
        args = parser.parse_args()

        # auto-detect settings from embedded __config__ if present
        data = np.load(args.filename)
        if '__config__' in data:
            cfg = json.loads(str(data['__config__']))
            if not args.coding     and 'coding'     in cfg: args.coding     = cfg['coding']
            if not args.mapping    and 'mapping'    in cfg: args.mapping    = cfg['mapping']
            if not args.colorspace and 'colorspace' in cfg: args.colorspace = cfg['colorspace']
            if not args.activation and 'activation' in cfg: args.activation = cfg['activation']
            args.model_size = cfg.get('model_size', 16)
            args.four       = cfg.get('four', False)
        else:
            args.model_size = 16
            args.four       = False

        names, sizes, values = self._load(data)
        self.precision = args.precision
        print(self._generate(args, names, sizes, values))

    def _load(self, data):
        names, sizes, values = [], {}, {}
        current = None
        for key in data.files:
            if key.startswith('__'):
                continue
            entry = data[key]
            name = key.replace(".", "_").upper()
            sizes[name] = list(entry.shape)
            values[name] = entry.flatten()
            if current:
                current.append(name)
                current = None
            else:
                current = [name]
                names.append(current)
        return names, sizes, values

    def _f(self, x, digits=None):
        d = digits if digits is not None else getattr(self, 'precision', 4)
        return f"{x:.{d}f}".replace("0.", ".")

    def _encode(self, args):
        L = args.coding

        if args.mapping == "polar":
            setup  = "\tfloat r = length(p) + 1e-8;"
            hint   = ["p.y/r", "p.x/r"]
            start  = 0
        elif args.mapping == "fourier":
            setup  = ""
            hint   = ["sin(PI * p.x)", "cos(PI * p.x)"]
            start  = 1
        else:  # legacy
            setup  = "\tfloat q = 22.;\n\tfloat ang = abs(fract(p.x*q)-.5)*2. * abs(fract(p.y*q)-.5)*2.;\n\tfloat lng = fract(length(p)*q);"
            hint   = ["ang", "lng"]
            start  = 0

        vals = ["p.x", "p.y"] + hint
        for i in range(start, start + L):
            freq = f"pow(2.0, {i}.) * PI"
            vals += [f"sin({freq}*p.x)", f"sin({freq}*p.y)",
                     f"cos({freq}*p.x)", f"cos({freq}*p.y)"]

        rows = "\n".join(
            f"\t\t{'  ' if r == 0 else ', '}{', '.join(vals[r*4:r*4+4])}"
            for r in range(4)
        )
        setup_block = f"\n{setup}" if setup else ""
        return f"const float PI = 3.14159265358979;\nmat4 encode(in vec2 p) {{{setup_block}\n\treturn mat4(\n{rows}\n\t);\n}}"

    # -------------------------------------------------------------------------
    # model_size = 16 — one mat4 per layer
    # -------------------------------------------------------------------------

    def _hidden_layer(self, out, a, w_vals, b_vals, fn):
        entries = []
        for i in range(16):
            bias = b_vals[i] if i < len(b_vals) else 0.0
            ws   = ", ".join(
                self._f(w_vals[i*16 + j*4 + k] if i*16 + j*4 + k < len(w_vals) else 0.0, 4)
                for j in range(4) for k in range(4)
            )
            entries.append(f"{fn}({self._f(bias)} + layeriate({a}, mat4({ws})))")
        joined = "\n\t\t, ".join(entries)
        return f"\t{out} = mat4(\n\t\t  {joined}\n\t);"

    def _output_layer(self, a, w_vals, b_vals, activation, colorspace, four=False):
        def channel(i):
            bias = b_vals[i] if i < len(b_vals) else 0.0
            dots = " + ".join(
                "dot({a}[{j}], vec4({ws}))".format(
                    a=a, j=j,
                    ws=", ".join(
                        self._f(w_vals[i*16 + j*4 + k] if i*16 + j*4 + k < len(w_vals) else 0.0)
                        for k in range(4)
                    )
                )
                for j in range(4)
            )
            expr = f"{self._f(bias)} + {dots}"
            if activation == "sine":
                return f"({expr}) * .5 + .5"
            else:
                return f"sigmoid({expr})"

        ch = [channel(i) for i in range(4)]
        return self._color_return(ch, colorspace, four)

    # -------------------------------------------------------------------------
    # model_size = 32 — two mat4s per layer (a0/a1, b0/b1)
    # -------------------------------------------------------------------------

    def _hidden_layer_32_first(self, out0, out1, a0, w_vals, b_vals, fn):
        """First hidden layer: input is one mat4 (16 inputs), output is two mat4s (32 outputs)."""
        entries = []
        for i in range(32):
            bias = b_vals[i] if i < len(b_vals) else 0.0
            # weight row i: 16 values (input is 16-wide)
            ws = ", ".join(
                self._f(w_vals[i*16 + j*4 + k] if i*16 + j*4 + k < len(w_vals) else 0.0, 4)
                for j in range(4) for k in range(4)
            )
            entries.append(f"{fn}({self._f(bias)} + layeriate({a0}, mat4({ws})))")
        joined0 = "\n\t\t, ".join(entries[:16])
        joined1 = "\n\t\t, ".join(entries[16:])
        return (
            f"\t{out0} = mat4(\n\t\t  {joined0}\n\t);\n"
            f"\t{out1} = mat4(\n\t\t  {joined1}\n\t);"
        )

    def _hidden_layer_32(self, out0, out1, a0, a1, w_vals, b_vals, fn):
        """Hidden layer: input is two mat4s (32 inputs), output is two mat4s (32 outputs)."""
        entries = []
        for i in range(32):
            bias = b_vals[i] if i < len(b_vals) else 0.0
            # weight row i: 32 values split into left (first 16) and right (last 16)
            ws_l = ", ".join(
                self._f(w_vals[i*32 + j*4 + k] if i*32 + j*4 + k < len(w_vals) else 0.0, 4)
                for j in range(4) for k in range(4)
            )
            ws_r = ", ".join(
                self._f(w_vals[i*32 + 16 + j*4 + k] if i*32 + 16 + j*4 + k < len(w_vals) else 0.0, 4)
                for j in range(4) for k in range(4)
            )
            entries.append(
                f"{fn}({self._f(bias)} + layeriate({a0}, mat4({ws_l})) + layeriate({a1}, mat4({ws_r})))"
            )
        joined0 = "\n\t\t, ".join(entries[:16])
        joined1 = "\n\t\t, ".join(entries[16:])
        return (
            f"\t{out0} = mat4(\n\t\t  {joined0}\n\t);\n"
            f"\t{out1} = mat4(\n\t\t  {joined1}\n\t);"
        )

    def _output_layer_32(self, a0, a1, w_vals, b_vals, activation, colorspace, four=False):
        """Output layer: input is two mat4s (32 inputs), output is vec3/vec4."""
        def channel(i):
            bias = b_vals[i] if i < len(b_vals) else 0.0
            # left half of weight row i (first 16)
            dots_l = " + ".join(
                "dot({a}[{j}], vec4({ws}))".format(
                    a=a0, j=j,
                    ws=", ".join(
                        self._f(w_vals[i*32 + j*4 + k] if i*32 + j*4 + k < len(w_vals) else 0.0)
                        for k in range(4)
                    )
                )
                for j in range(4)
            )
            # right half of weight row i (last 16)
            dots_r = " + ".join(
                "dot({a}[{j}], vec4({ws}))".format(
                    a=a1, j=j,
                    ws=", ".join(
                        self._f(w_vals[i*32 + 16 + j*4 + k] if i*32 + 16 + j*4 + k < len(w_vals) else 0.0)
                        for k in range(4)
                    )
                )
                for j in range(4)
            )
            expr = f"{self._f(bias)} + {dots_l} + {dots_r}"
            if activation == "sine":
                return f"({expr}) * .5 + .5"
            else:
                return f"sigmoid({expr})"

        ch = [channel(i) for i in range(4)]
        return self._color_return(ch, colorspace, four)

    # -------------------------------------------------------------------------
    # shared output formatting
    # -------------------------------------------------------------------------

    def _color_return(self, ch, colorspace, four=False):
        color_decode = ""
        if colorspace == "ycbcr":
            color_decode = "\tcolor = mat3(1, 0, 1.402, 1, -.344136, -.714136, 1, 1.772, 0.) * color;"
        elif colorspace == "yuv":
            color_decode = "\tcolor = mat3(1, 0, 1.13983, 1, -.39465, -.58060, 1, 2.03211, 0) * color;"

        if four:
            # 4th channel encodes a luminance ratio — apply it
            grayed = "\tfloat grayed = kolor.a / ((color.r + color.g + color.b) / 3.);\n\tcolor *= grayed;"
            alpha_line = f"\t\t, {ch[3]}\n"
            after_decode = f"{grayed}\n"
        else:
            alpha_line = ""
            after_decode = ""

        kolor_type = "vec4" if four else "vec3"
        return (
            f"\t{kolor_type} kolor = {kolor_type}(\n"
            f"\t\t  {ch[0]}\n"
            f"\t\t, {ch[1]}\n"
            f"\t\t, {ch[2]}\n"
            f"{alpha_line}"
            f"\t);\n"
            f"\tvec3 color = kolor.rgb;\n"
            + (f"{color_decode}\n" if color_decode else "")
            + after_decode
            + f"\treturn color;"
        )

    # -------------------------------------------------------------------------
    # generate
    # -------------------------------------------------------------------------

    def _generate(self, args, names, sizes, values):
        use_sine   = args.activation == "sine"
        model_size = getattr(args, 'model_size', 16)

        helpers = (
            "float relu(float f)   { return max(.0, f); }\n"
            "float sine0(float f)  { return sin(30.0 * f); }\n"
            "float sine(float f)   { return sin(f); }\n"
            + ("" if use_sine else "float sigmoid(float f) { return 1. / (1. + exp(-f)); }\n")
            + "float layeriate(mat4 lair, mat4 q) {\n"
            "\treturn dot(lair[0], q[0]) + dot(lair[1], q[1])\n"
            "\t     + dot(lair[2], q[2]) + dot(lair[3], q[3]);\n"
            "}"
        )

        encode = self._encode(args)

        if model_size == 32:
            nn = self._generate_nn_32(args, names, values, use_sine)
        else:
            nn = self._generate_nn_16(args, names, values, use_sine)

        main = (
            "void mainImage(out vec4 to, in vec2 at) {\n"
            "\tvec2 current_rz = iResolution.xy / 2048.;\n"
            "\tvec2 old_rz = texelFetch(iChannel0, ivec2(0), 0).xy;\n"
            "\tif (distance(current_rz, old_rz) < .11) {\n"
            "\t\tto = texelFetch(iChannel0, ivec2(at), 0);\n"
            "\t} else {\n"
            "\t\tivec2 i = ivec2(at);\n"
            "\t\tif (0 == i.x && 0 == i.y) {\n"
            "\t\t\tto.xy = current_rz;\n"
            "\t\t} else {\n"
            "\t\t\tvec2 uv = (at * 2. - iResolution.xy) / iResolution.y * vec2(1., -1.);\n"
            "\t\t\tto = vec4(nn(uv), 1.);\n"
            "\t\t}\n"
            "\t}\n"
            "}"
        )

        return "\n\n".join([helpers, encode, nn, main])

    def _generate_nn_16(self, args, names, values, use_sine):
        bufs = ["l1", "l2"]
        body = ["\tmat4 l2, l1 = encode(p);"]
        for idx, (w_name, b_name) in enumerate(names):
            a   = bufs[idx % 2]
            out = bufs[(idx + 1) % 2]
            w, b = values[w_name], values[b_name]
            last = idx + 1 == len(names)
            if last:
                body.append(self._output_layer(a, w, b, args.activation, args.colorspace, getattr(args, 'four', False)))
            else:
                fn = ("sine0" if idx == 0 else "sine") if use_sine else "relu"
                body.append(self._hidden_layer(out, a, w, b, fn))
        return "vec3 nn(vec2 p) {\n" + "\n".join(body) + "\n}"

    def _generate_nn_32(self, args, names, values, use_sine):
        # ping-pong between (l1a,l1b) and (l2a,l2b)
        bufs = [("l1a", "l1b"), ("l2a", "l2b")]
        body = ["\tmat4 l2a, l2b, l1b, l1a = encode(p);"]
        for idx, (w_name, b_name) in enumerate(names):
            a0, a1   = bufs[idx % 2]
            out0, out1 = bufs[(idx + 1) % 2]
            w, b = values[w_name], values[b_name]
            last = idx + 1 == len(names)
            fn = ("sine0" if idx == 0 else "sine") if use_sine else "relu"
            if last:
                body.append(self._output_layer_32(a0, a1, w, b, args.activation, args.colorspace, getattr(args, 'four', False)))
            elif idx == 0:
                # first hidden layer: input is single mat4 (16-wide encoding)
                body.append(self._hidden_layer_32_first(out0, out1, a0, w, b, fn))
            else:
                body.append(self._hidden_layer_32(out0, out1, a0, a1, w, b, fn))
        return "vec3 nn(vec2 p) {\n" + "\n".join(body) + "\n}"


if __name__ == "__main__":
    Translate().main()

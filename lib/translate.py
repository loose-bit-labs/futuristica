import argparse
import numpy as np


class Translate:
    def main(self):
        parser = argparse.ArgumentParser(description="Translate weights to GLSL")
        parser.add_argument('filename')
        parser.add_argument("-c", "--coding",     type=int, default=3)
        parser.add_argument("-e", "--mapping",    choices=["polar", "fourier", "legacy"], default="polar")
        parser.add_argument("-k", "--colorspace", choices=["rgb", "ycbcr", "yuv"], default="ycbcr")
        parser.add_argument("-a", "--activation", choices=["sine", "relu"], default="sine")
        args = parser.parse_args()

        names, sizes, values = self._load(args.filename)
        print(self._generate(args, names, sizes, values))

    def _load(self, filename):
        data = np.load(filename)
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

    def _f(self, x, digits=2):
        return f"{x:.{digits}f}".replace("0.", ".")

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

    def _output_layer(self, a, w_vals, b_vals, activation, colorspace):
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
        color_decode = ""
        if colorspace == "ycbcr":
            color_decode = "\tcolor = mat3(1, 0, 1.402, 1, -.344136, -.714136, 1, 1.772, 0.) * color;"
        elif colorspace == "yuv":
            color_decode = "\tcolor = mat3(1, 0, 1.13983, 1, -.39465, -.58060, 1, 2.03211, 0) * color;"

        grayed = "\tfloat grayed = kolor.a / ((color.r + color.g + color.b) / 3.);\n\tcolor *= grayed;"

        return (
            f"\tvec4 kolor = vec4(\n"
            f"\t\t  {ch[0]}\n"
            f"\t\t, {ch[1]}\n"
            f"\t\t, {ch[2]}\n"
            f"\t\t, {ch[3]}\n"
            f"\t);\n"
            f"\tvec3 color = kolor.rgb;\n"
            + (f"{color_decode}\n" if color_decode else "")
            + f"{grayed}\n"
            f"\treturn color;"
        )

    def _generate(self, args, names, sizes, values):
        use_sine = args.activation == "sine"

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

        bufs = ["l1", "l2"]
        nn_body = ["\tmat4 l2, l1 = encode(p);"]
        for idx, (w_name, b_name) in enumerate(names):
            a   = bufs[idx % 2]
            out = bufs[(idx + 1) % 2]
            w, b = values[w_name], values[b_name]
            last = idx + 1 == len(names)
            if last:
                nn_body.append(self._output_layer(a, w, b, args.activation, args.colorspace))
            else:
                fn = ("sine0" if idx == 0 else "sine") if use_sine else "relu"
                nn_body.append(self._hidden_layer(out, a, w, b, fn))

        nn = "vec3 nn(vec2 p) {\n" + "\n".join(nn_body) + "\n}"

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


if __name__ == "__main__":
    Translate().main()

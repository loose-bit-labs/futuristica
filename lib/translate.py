#!/usr/bin/env python
#############################################################################

import argparse
import numpy as np

## 
 #
 # Takes a generated weight file and generates the GLSL
 # code for shadertoy. 
 #
 # If you are using 16 model_size use -s to generate the 
 # mat4 version which runs much nicer
 #
 ##
class Translate:
    def main(self):
        parser = argparse.ArgumentParser(description="Translate weights to GLSL")
        parser.add_argument('filename')
        parser.add_argument("-c", "--coding",     type=int, default=3)
        parser.add_argument("-e", "--mapping",    choices=["polar", "fourier", "legacy"], default="polar")
        parser.add_argument("-k", "--colorspace", choices=["rgb", "ycbcr", "yuv"], default="ycbcr")
        parser.add_argument("-a", "--activation", choices=["sine", "relu"], default="sine")

        args = parser.parse_args()

        print("float relu(float f)  { return max(.0, f); }")
        print("float sine0(float f) { return sin(30.0 * f); }  // first layer omega_0=30")
        print("float sine(float f)  { return sin(f); }          // hidden layers omega_0=1")
        if args.activation == "relu":
            print("float sigmoid(float f) { return 1. / (1. + exp(-f)); }")
        print("float layeriate(mat4 lair, mat4 q) {")
        print("\treturn dot(lair[0], q[0]) + dot(lair[1], q[1]) + dot(lair[2], q[2]) + dot(lair[3], q[3]);")
        print("}")
        print("");

        data, workspace_size, names, sizes, values = self.load(args)

        self.is_four = True
        self.my_sweetie(args, data, workspace_size, names, sizes, values)

        self.make_main()
    #end of main


    def color_back(self, args):
        if "ycbcr" == args.colorspace:
            print("\tcolor = mat3(1, 0, 1.402, 1, -0.344136, -0.714136, 1, 1.772, 0.) * color;")
        elif "yuv" == args.colorspace:
            print("\tcolor = mat3(1, 0, 1.13983, 1, -0.39465, -0.58060, 1, 2.03211, 0) * color;")
        if self.is_four:
            print("\tfloat grayed = kolor.a / ((color.r + color.g + color.b ) / 3.);")
            print("\tcolor *= grayed;")
        print("\treturn color;")
    # end of color_back



    def load(self, args):
        data = np.load(args.filename);
        workspace_size = 1
        names = []
        sizes = {}
        values = {}

        current = None
        for key in data.files:
            entry = data[key]
            name = key.replace(".", "_").upper()
            sizes[name] = []
            values[name] = entry.flatten()
            if current:
                current.append(name)
                current = None
            else:
                current = [name]
                names.append(current)
            n = 1
            for size in entry.shape:
                n = n * size
                sizes[name].append(size);
                workspace_size = max(workspace_size, size)
        return data, workspace_size, names, sizes, values
    #end of load


    def my_sweetie(self, args, data, workspace_size, names, sizes, values):
        # encode() mirrors positional_encoding() in futuristica.py.
        # All three mappings produce the same 16-wide mat4 for L=3.
        # Slots 0-1: always [x, y]
        # Slots 2-3: the spatial hint — differs per --mapping
        # Slots 4+:  Fourier sin/cos bands
        print("const float PI = 3.14159265358979;")
        print("mat4 encode(in vec2 p) {")

        L = args.coding  # number of Fourier bands

        if args.mapping == "polar":
            # sin/cos of angle — globally continuous, no atan2 discontinuity
            print("\t// polar: sin/cos of angle, globally continuous")
            print("\tfloat r = length(p) + 1e-8;")
            hint = "p.y/r, p.x/r"  # sin(θ), cos(θ)
            bands_start = 0

        elif args.mapping == "fourier":
            # pure Fourier — first band fills hint slots, loop starts at 1
            print("\t// fourier: pure spectral, no spatial hint")
            hint = "sin(PI * p.x), cos(PI * p.x)"
            bands_start = 1

        else:  # legacy
            # original abs(fract()) kink-based encoding — for reproducibility
            print("\t// legacy: abs(fract()) kink encoding")
            print("\tfloat q = 22.;")
            print("\tfloat ang = abs(fract(p.x*q)-.5)*2. * abs(fract(p.y*q)-.5)*2.;")
            print("\tfloat lng = fract(length(p)*q);")
            hint = "ang, lng"
            bands_start = 0

        # build the mat4 rows: 16 values total, 4 per row
        vals = ["p.x", "p.y"] + hint.split(", ")
        for i in range(bands_start, bands_start + L):
            freq = f"pow(2.0, {i}.) * PI"
            vals += [f"sin({freq}*p.x)", f"sin({freq}*p.y)",
                     f"cos({freq}*p.x)", f"cos({freq}*p.y)"]

        print("\treturn mat4(")
        for row in range(4):
            chunk = vals[row*4 : row*4+4]
            sep = "" if row == 3 else ","
            print(f"\t\t{', '.join(chunk)}{sep}")
        print("\t);\n}\n")

        print(f"vec3 nn(vec2 p) " + "{")
        print("\tmat4 l2, l1 = encode(p);")

        a = "l1"
        b = "l2"
        for index, pair in enumerate(names):
            weights = pair[0]
            bias    = pair[1]
            last = index + 1 == len(names)

            values_weights = values[weights]
            values_bias    = values[bias]
            layer_size     = sizes[weights]

            print(f"\t// layer {index}: {sizes[weights]} {last}");
            if last:
                if self.is_four:
                    # bugfix: was emitting values_bias[2] twice; 4th channel needs [3]
                    if args.activation == "sine":
                        def wrap(b, d): return f"({b:.2f}{d}) * .5 + .5".replace("0.", ".")
                    else:
                        def wrap(b, d): return f"sigmoid({b:.2f}{d})".replace("0.", ".")
                    print("\tvec4 kolor = vec4(")
                    print(f"\t\t  {wrap(values_bias[0], self.dodod(a, values_weights, 0))}")
                    print(f"\t\t, {wrap(values_bias[1], self.dodod(a, values_weights, 1))}")
                    print(f"\t\t, {wrap(values_bias[2], self.dodod(a, values_weights, 2))}")
                    print(f"\t\t, {wrap(values_bias[3], self.dodod(a, values_weights, 3))}")
                    print("\t);")
                    print("\tvec3 color = kolor.rgb;")
                else:
                    if args.activation == "sine":
                        def wrap3(b, d): return f"({b:.2f}{d}) * .5 + .5".replace("0.", ".")
                    else:
                        def wrap3(b, d): return f"sigmoid({b:.2f}{d})".replace("0.", ".")
                    print("\tvec3 color = vec3(")
                    print(f"\t\t  {wrap3(values_bias[0], self.dodod(a, values_weights, 0))}")
                    print(f"\t\t, {wrap3(values_bias[1], self.dodod(a, values_weights, 1))}")
                    print(f"\t\t, {wrap3(values_bias[2], self.dodod(a, values_weights, 2))}")
                    print("\t);")
            else:
                print(f"\t{b} = mat4(")
                c = " "
                for i in range(16):
                    if i < len(values_bias):
                        bias = values_bias[i]
                    else:
                        bias = 0
                    # First layer gets omega_0=30, all hidden layers get omega_0=1
                    if args.activation == "sine":
                        fn = "sine0" if index == 0 else "sine"
                    else:
                        fn = "relu"
                    s = f"{fn}({bias:.2f}".replace("0.", ".")
                    s += self.dodod_jr(a, values_weights, i)
                    s += ")"
                    print(f"\t\t{c} {s}") # // {i}")
                    c = ","
                print("\t);")
            t = a
            a = b
            b = t
        
        self.color_back(args)
        print("}")
    # end of my_sweetie (just in code)

    def dodod(self, a, weights, i):
        s = ""
        for j in range(4):
            s += f" + dot({a}[{j}], vec4("
            cc = ""
            for k in range(4):
                index = i * 16 + j * 4 + k
                if index < len(weights):
                    weight = weights[index]
                else:
                    weight = 0
                s += f"{cc}{weight:.2f}".replace("0.", ".")
                cc = ","
            s += "))"
        return s
    # end of dodo

    def dodod_jr(self, a, weights, i):
        s = f" + layeriate({a}, mat4("
        cc = ""
        for j in range(4):
            for k in range(4):
                index = i * 16 + j * 4 + k
                if index < len(weights):
                    weight = weights[index]
                else:
                    weight = 0
                s += f"{cc}{weight:.4f}".replace("0.", ".")
                cc = ","
        s += "))"
        return s
    # end of dodo_jr

    def make_main(self):
        print("void mainImage(out vec4 to, in vec2 at) {")
        print("\tvec2 current_rz = iResolution.xy / 2048.;")
        print("\tvec2 old_rz = texelFetch(iChannel0, ivec2(0), 0 ).xy;")
        print("\tfloat d = distance(current_rz, old_rz);")
        print("\tif (d<.11) {")
        print("\t\tto =  texelFetch(iChannel0, ivec2(at), 0 );")
        print("\t} else {")
        print("\t\tivec2 i = ivec2(at);")
        print("\t\tif (0 == i.x && 0 == i.y) {")
        print("\t\t\tto.xy = current_rz;")
        print("\t\t} else {")
        print("\t\t\tvec2 uv = (at * 2. - iResolution.xy) / iResolution.y * vec2(1.,-1.);")
        print("\t\t\tto = vec4(nn(uv), 1.);")
        print("\t\t}")
        print("\t }")
        print("}")
    # end of make_main
        

# end-of class Translate

if __name__ == "__main__":
    Translate().main()

# EOF
#############################################################################

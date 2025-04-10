#!/usr/bin/env python
#############################################################################

import argparse
import numpy as np

class Translate:
    def main(self):
        parser = argparse.ArgumentParser(description="Translate weights to GLSL")
        parser.add_argument('filename')
        parser.add_argument("-c", "--coding", type=int, default=3)
        parser.add_argument("-p", "--polar",  action="store_true")
        parser.add_argument("-s", "--sweet",  action="store_true")
        parser.add_argument("-k", "--colorspace",  choices=["rgb", "ycbcr", "yuv"], default="ycbcr")

        args = parser.parse_args()

        print("float relu(float f) { return max(.0, f);}")
        print("float sigmoid(float f) { return 1. / (1. + exp(-f)); }")
        print("");

        data, workspace_size, names, sizes, values = self.load(args)

        if args.sweet and self.sweet16(sizes):
            self.my_sweetie(args, data, workspace_size, names, sizes, values)
        else:
            #self.arrrrrrrr(args, data, workspace_size, names, sizes, values)
            self.innie(args, data, workspace_size, names, sizes, values)


        print("void mainImage(out vec4 to, in vec2 at) {")
        print("\tvec2 uv = (at * 2. - iResolution.xy)/iResolution.y * vec2(1.,-1.);")
        print("\tif (iFrame<33) to = vec4(nn(uv), 1.); else discard; // 🤖")
        print("}")
    #end of main


    # this version uses arrays and for loops
    def arrrrrrrr(self, args, data, workspace_size, names, sizes, values):
        for name,v in sizes.items():
            for i, size in enumerate(v):
                print(f"const int {name}_{i}_SIZE = {size};");
        print("")
        
        for name, valuez in values.items():
            vv = ",".join(map(lambda x: "{:.2f}".format(x).replace("0.", "."),valuez)) # entry.flatten()))
            print(f"const float {name}[] = float[{len(valuez)}]({vv});")
        print("");
        
        if args.coding > 0:
            self.encodo(args.coding, args.polar)
            print("");

        w1 = "WK1"
        w2 = "WK2"

        src = "p";
        dst = w1

        workspace = ",".join([".0"] * workspace_size)
        for i in range(0,2):
            print(f"float WK{i+1}[{workspace_size}];"); # = float[{workspace_size}]({workspace});");

        print(f"vec3 nn(vec2 {src}) " + "{")
        print("\tvec3 color = vec3(.0);")

        if args.coding > 0:
            print(f"\tpositionalEncoding({src}, encoded);");
            src = "encoded"
        
        index = 0
        for pair in names:
            weights = pair[0]
            bias    = pair[1]
            index   = 1 + index
            last = index == len(names)

            print(f"\tfor (int i = 0; i < {weights}_0_SIZE ; i++) " + "{")
            print(f"\t\tfloat sum = {bias}[i];")
            print(f"\t\tfor (int j = 0; j < {weights}_1_SIZE ; j++) " + "{")
            print(f"\t\t\tsum += {src}[j] * {weights}[i * {weights}_1_SIZE + j];")
            print( "\t\t}")
            if last:
                dst = "color"
                print(f"\t\t{dst}[i] = sigmoid(sum);")
            else:
                print(f"\t\t{dst}[i] = relu(sum);");
                if src != w1:
                    src = w1
                    dst = w2
                else:
                    dst = w1
                    src = w2
                #src = "WK"
            print( "\t}")
        self.color_back(args)
        print("}")
    # end of arrrrrrrr


    def color_back(self, args):
        if "ycbcr" == args.colorspace:
            print("\treturn mat3(1, 0, 1.402, 1, -0.344136, -0.714136, 1, 1.772, 0.) * color;")
        elif "yuv" == args.colorspace:
            print("\treturn mat3(1, 0, 1.13983, 1, -0.39465, -0.58060, 1, 2.03211, 0) * color;")
        else:
            print("\treturn color;")
    # end of color_back

        
    def encodo(self, coding=3, polar=False, got_squares=True):
        print(f"float encoded[{2 + 4 * coding}];")
        print("void positionalEncoding(vec2 coords, out float encoded[16]) {")
        print("\t// Index for encoded array")
        print("\tint idx = 0;")
        print("")
        print("\t// Store the original coordinates")
        print("\tencoded[idx++] = coords.x;")
        print("\tencoded[idx++] = coords.y;")
        print("")
        if polar:
            print("\t// Compute angle and length")
            print("\t//float angle = atan(coords.y, coords.x);")
            print("\tfloat q = 22.;")
            print("\tfloat angle = abs(fract(coords.x * q)-.5) *2. * abs(fract(coords.y * q)-.5) *2.;")
            print("\tfloat length = fract(length(coords) * q);")
            print("")
            print("\tencoded[idx++] = angle;")
            print("\tencoded[idx++] = length;")
            print("")
        print(f"\t// Compute sinusoidal encodings for L = {coding}")
        print(f"\tfor (int i = 0; i < {coding}; i++) " + "{")
        t = ""
        if got_squares:
            print("\t\tif(1 == i %2) {")
            t = "\t"
        print(f"{t}\t\t\t// Curvy shapes")
        print(f"{t}\t\t\tencoded[idx++] = sin((pow(2.0, float(i)) * 3.141592653589793) * coords.x);")
        print(f"{t}\t\t\tencoded[idx++] = sin((pow(2.0, float(i)) * 3.141592653589793) * coords.y);")
        print(f"{t}\t\t\tencoded[idx++] = cos((pow(2.0, float(i)) * 3.141592653589793) * coords.x);")
        print(f"{t}\t\t\tencoded[idx++] = cos((pow(2.0, float(i)) * 3.141592653589793) * coords.y);")
        if got_squares:
            print("\t\t} else {")
            print("\t\t\t// Straight edges")
            print("\t\t\tfloat scale1 = pow(2.0, float(i)) - 0.5;")
            print("\t\t\tfloat scale2 = pow(2.0, float(i) + 0.5) - 0.5;")
            print("\t\t\tencoded[idx++] = abs(scale1 * coords.x);")
            print("\t\t\tencoded[idx++] = abs(scale1 * coords.y);")
            print("\t\t\tencoded[idx++] = abs(scale2 * coords.x);")
            print("\t\t\tencoded[idx++] = abs(scale2 * coords.y);")
            print("\t\t}")
        print("\t}")
        print("}")
    # end of encodo


    def old_encodo(self, coding=3, polar=False, got_squares=True):
        print(f"float encoded[{2 + 4 * coding}];")
        print(f"void positionalEncoding(vec2 p) " + "{")
        print("\tint index = 0;")
        print("\t")
        print("\t// Store original coordinates")
        print("\tencoded[index++] = p.x;")
        print("\tencoded[index++] = p.y;")
        print("\t")

        if polar:
            print("\tencoded[index++] = atan(p.y,p.x);")
            print("\tencoded[index++] = length(p);")

        print(f"\tfor (int i = 0; i < {coding} ; i++) " + "{")
        print("\t\tfloat freq = exp2(float(i)) * 3.14159265359;")
        t = ""
        if got_squares:
            print("\t\tif(1 == i %2) {")
            t = "\t"
        print(f"{t}\t\tencoded[index++] = sin(freq * p.x);")
        print(f"{t}\t\tencoded[index++] = sin(freq * p.y);")
        print(f"{t}\t\tencoded[index++] = cos(freq * p.x);")
        print(f"{t}\t\tencoded[index++] = cos(freq * p.y);")
        if got_squares:
            print("\t\t} else {")
            print("\t\t\tfloat j = float(i)+.5;")
            print("\t\t\tencoded[index++] = abs((exp2(float(i)) - 0.5) * p.x);")
            print("\t\t\tencoded[index++] = abs((exp2(float(i)) - 0.5) * p.y);")
            print("\t\t\tencoded[index++] = abs((exp2(float(j)) - 0.5) * p.x);")
            print("\t\t\tencoded[index++] = abs((exp2(float(j)) - 0.5) * p.y);")
            print("\t\t}")
        print("\t}")
        print("}\n")
    # end of old_encodo


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


    def innie(self, args, data, workspace_size, names, sizes, values):
        if args.coding > 0:
            self.encodo(args.coding, args.polar)
            print("");

        workspace = ",".join([".0"] * workspace_size)
        for i in range(0,2):
            print(f"float WK{i+1}[{workspace_size}];"); # = float[{workspace_size}]({workspace});");

        w1 = "WK1"
        w2 = "WK2"

        src = "p"
        dst = w1

        print(f"vec3 nn(vec2 {src}) " + "{")
        print("\tvec3 color = vec3(.0);")

        if args.coding > 0:
            print(f"\tpositionalEncoding({src}, encoded);");
            src = "encoded"
        
        index = 0
        fnk = "relu"

        for pair in names:
            weights = pair[0]
            bias    = pair[1]
            index   = 1 + index
            last = index == len(names)

            values_weights = values[weights]
            values_bias    = values[bias]
            layer_size     = sizes[weights]

            print(f"\t// layer {index}: {sizes[weights]}");

            if last:
                dst = "color"
                fnk = "sigmoid"

            for i in range(layer_size[0]):
                tmp = f"{dst}[{i}]"
                q = f"{values_bias[i]:.2f}"
                s = f"\t{tmp:7s} = {fnk}({q}"
                for j in range(layer_size[1]):
                    k = i * layer_size[1] + j
                    s += f" + {src}[{j}]*{values_weights[k]:.2f}"
                print(f"{s.replace('0.', '.')});");

            if last:
                dst = "color"
            else:
                if src != w1:
                    src = w1
                    dst = w2
                else:
                    dst = w1
                    src = w2
        self.color_back(args)
        print("}")
    # end of innie


    def sweet16(self, sizes):
        values = list(sizes.values())
        last_two_values = values[-2:]
        if last_two_values != [[3, 16], [3]]:
            return False
        for value in values[:-2]:
            if value not in [[16, 16], [16]]:
                return False
        return True
    # end of sweet16


    def my_sweetie(self, args, data, workspace_size, names, sizes, values):
        print("float encF(in float f) { return exp2(f) * 3.14159265359; }")
        print("float encS(in float f, in float g, float v) { return abs((exp2(f + g) - 0.5) * v); }");
        print("")

        print("mat4 encode(in vec2 p) {")
        print("\tfloat q = 22.;")
        print("\tfloat angle = abs(fract(p.x * q)-.5) *2. * abs(fract(p.y * q)-.5) *2.;")
        print("\tfloat lengt = fract(length(p) * q);")
        print("\treturn mat4(")
        print("\t\tp.x, p.y, angle, lengt,")
        #print("\t\tp.x, p.y, atan(p.y, p.x), length(p),")
        c = ","
        for i in range(0,3):
            f = f"encF({i}.)"
            s1 = f"encS({i}., .0)"
            s2 = f"encS({i}., .5)"
            if 1 == i % 2:
                print(f"\t\tsin({f} * p.x),     sin({f} * p.y),")
                print(f"\t\tcos({f} * p.x),     cos({f} * p.y),")
            else:
                print(f"\t\tencS({i}., .0, p.x), encS({i}., .0, p.y),")
                print(f"\t\tencS({i}., .5, p.x), encS({i}., .5, p.y){c}")
                c = ""
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
                print("\tvec3 color = vec3(")
                print(f"\t\t  sigmoid({values_bias[0]:.2f}{self.dodod(a, values_weights, 0)})")
                print(f"\t\t, sigmoid({values_bias[1]:.2f}{self.dodod(a, values_weights, 1)})")
                print(f"\t\t, sigmoid({values_bias[2]:.2f}{self.dodod(a, values_weights, 2)})")
                print("\t);")
            else:
                print(f"\t{b} = mat4(")
                c = " "
                for i in range(16):
                    s = f"relu({values_bias[i]:.2f}".replace("0.", ".")
                    s += self.dodod(a, values_weights, i)
                    s += ")"
                    print(f"\t\t{c} {s}") # // {i}")
                    c = ","
                print("\t);")
            t = a
            a = b
            b = t
        
        self.color_back(args)
        print("}")
    #end of my_sweetie

    def dodod(self, a, values_weights, i):
        s = ""
        for j in range(4):
            s += f" + dot({a}[{j}], vec4("
            cc = ""
            for k in range(4):
                s += f"{cc}{values_weights[i * 16 + j * 4 + k]:.2f}".replace("0.", ".")
                cc = ","
            s += "))"
        return s

        

# end-of class Translate

if __name__ == "__main__":
    Translate().main()

# EOF
#############################################################################

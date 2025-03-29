#!/usr/bin/env python
#############################################################################

import argparse
import numpy as np

class Translate:
    def main(self):
        parser = argparse.ArgumentParser(description="Translate weights to GLSL")
        parser.add_argument('filename')
        parser.add_argument("--coding", type=int, default=2)

        args = parser.parse_args()

        data = np.load(args.filename);

        for key in data.files:
            entry = data[key]
            name = key.replace(".", "_").upper()
            if "BIAS" in name:
                #print(f"const int {name}_SIZE = {size};");
                pass 
            else:
                i = 0
                n = 1
                for size in entry.shape:
                    n = n * size
                    print(f"const int {name}_{i}_SIZE = {size};");
                    i = i + 1
        print("")

        workspace_size = 1

        names = []
        current = None
        for key in data.files:
            entry = data[key]
            name = key.replace(".", "_").upper()
            if current:
                current.append(name)
                current = None
            else:
                current = [name]
                names.append(current)
            n = 1
            for size in entry.shape:
                n = n * size
                workspace_size = max(workspace_size, size)
            values = ",".join(map(lambda x: "{:.2f}".format(x).replace("0.", "."), entry.flatten()))
            print(f"const float {name}[] = float[{n}]({values});");
        print("")

        workspace = ",".join([".0"] * workspace_size)
        for i in range(0,2):
            print(f"float WORKSPACE{i+1}[{workspace_size}];"); # = float[{workspace_size}]({workspace});");

        if args.coding > 0:
            print(f"float encoded[{2 + 4 * args.coding}];")

        print("");
        
        w1 = "WORKSPACE1"
        w2 = "WORKSPACE2"

        src = "p";
        dst = w1

        got_squares = True
        if args.coding > 0:
            print(f"void encoder(vec2 {src}) " + "{")
            print("\tint index = 0;")
            print("\t")
            print("\t// Store original coordinates")
            print("\tencoded[index++] = p.x;")
            print("\tencoded[index++] = p.y;")
            print("\t")
            print(f"\tfor (int i = 0; i < {args.coding} ; i++) " + "{")
            print("\t\tfloat freq = exp2(float(i)) * 3.14159265359;")
            if got_squares:
                print("\t\tif(1 == i %2) {")
            print("\t\tencoded[index++] = sin(freq * p.x);")
            print("\t\tencoded[index++] = sin(freq * p.y);")
            print("\t\tencoded[index++] = cos(freq * p.x);")
            print("\t\tencoded[index++] = cos(freq * p.y);")
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

        print(f"vec3 nn(vec2 {src}) " + "{")
        print("\tvec3 color = vec3(.0);")

        if args.coding > 0:
            print(f"\tencoder({src});");
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
                print(f"\t\t{dst}[i] = sum; // 1.0 / (1.0 + exp(-sum)); // Sigmoid activation")
            else:
                print(f"\t\t{dst}[i] = max(sum, 0.0); // ReLU activation") #  last needs to be sigmoid....
                if src != w1:
                    src = w1
                    dst = w2
                else:
                    dst = w1
                    src = w2
                #src = "WORKSPACE"
            print( "\t}")
        print("\treturn color;")
        print("}")

        print("");
        print("void mainImage(out vec4 to, in vec2 at) {");
        print("\tvec2 uv = (at * 2. - iResolution.xy)/iResolution.y * vec2(1.,-1.);");
        print("\tto = vec4(nn(uv), 1.); // 🤖");
        print("}");
# end-of class Translate

if __name__ == "__main__":
    Translate().main()

# EOF
#############################################################################

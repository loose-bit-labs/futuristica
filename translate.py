#!/usr/bin/env python
#############################################################################

import argparse
import numpy as np


class Translate:
    def main(self):
        parser = argparse.ArgumentParser(description="Translate weights to GLSL")
        parser.add_argument('filename')
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

        maxo = 1

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
            if n > maxo:
                maxo =n
            values = ",".join(map(lambda x: "{:.2f}".format(x).replace("0.", "."), entry.flatten()))
            #print(",".join(map(lambda x: "{:.2f}".format(x).replace("0.", "."), entry.flatten())))
            print(f"const float {name}[] = float[{n}]({values});");
            #print(",".join(map("{:.4f}".format, entry.flatten())))
            #print(f");")
        print("")

        workspace = ",".join([".0"] * maxo)
        for i in range(0,2):
            print(f"float WORKSPACE{i+1}[] = float[{maxo}]({workspace});");

        print("");

        print("vec3 mlp(vec2 x) {")
        print("\tvec3 q = vec3(.0);")

        src = "x";
        dst = "WORKSPACE1"

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
                dst = "q"
                print(f"\t\t{dst}[i] = 1.0 / (1.0 + exp(-sum)); // Sigmoid activation")
            else:
                print(f"\t\t{dst}[i] = max(sum, 0.0); // ReLU activation") #  last needs to be sigmoid....
                if src != "WORKSPACE1":
                    src = "WORKSPACE1"
                    dst = "WORKSPACE2"
                else:
                    dst = "WORKSPACE1"
                    src = "WORKSPACE2"
                #src = "WORKSPACE"
            print( "\t}")
        print("\treturn q;")
        print("}")

        print("");
        print("void mainImage(out vec4 to, in vec2 at) {");
        print("\tvec2 uv = (at * 2. - iResolution.xy)/iResolution.y * vec2(1.,-1.);");
        print("\tto = vec4(mlp(uv), 1.); // 🤖");
        print("}");
# end-of class Translate

if __name__ == "__main__":
    Translate().main()

# EOF
#############################################################################

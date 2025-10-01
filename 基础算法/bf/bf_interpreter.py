def run_bf(code):
    tape = [0] * 30000
    ptr = 0
    i = 0
    output = ""
    while i < len(code):
        c = code[i]
        if c == ">":
            ptr += 1
        elif c == "<":
            ptr -= 1
        elif c == "+":
            tape[ptr] = (tape[ptr] + 1) % 256
        elif c == "-":
            tape[ptr] = (tape[ptr] - 1) % 256
        elif c == ".":
            output += chr(tape[ptr])
        elif c == ",":
            try:
                s = input("Input a char: ")
                if s == "":  # 用户直接回车，表示EOF
                    tape[ptr] = 0
                else:
                    tape[ptr] = ord(s[0])
            except EOFError:  # Ctrl-D (macOS/Linux) 或 Ctrl-Z (Windows) 触发
                tape[ptr] = 0
        elif c == "[":
            if tape[ptr] == 0:
                open_brackets = 1
                while open_brackets:
                    i += 1
                    if code[i] == "[":
                        open_brackets += 1
                    elif code[i] == "]":
                        open_brackets -= 1
        elif c == "]":
            if tape[ptr] != 0:
                close_brackets = 1
                while close_brackets:
                    i -= 1
                    if code[i] == "[":
                        close_brackets -= 1
                    elif code[i] == "]":
                        close_brackets += 1
                i -= 1
        i += 1
    print(output)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bf_interpreter.py yourcode.bf")
    else:
        with open(sys.argv[1]) as f:
            code = "".join(ch for ch in f.read() if ch in "<>+-.,[]")
        run_bf(code)

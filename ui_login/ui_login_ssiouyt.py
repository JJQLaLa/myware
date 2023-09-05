#  *_* coding:utf8 *_*
import tkinter
from functools import partial


# 按钮输入调用
def get_input(entry, argu):
    # 从entry窗口展示中获取输入的内容
    input_data = entry.get()
    # 出现连续+，则第二个+为无效输入，不做任何处理
    if (input_data[-1:] == '+') and (argu == '+'):
        return
    # 出现连续+--，则第三个-为无效输入，不做任何处理
    if (input_data[-2:] == '+-') and (argu == '-'):
        return
    # 窗口已经有--后面字符不能为+或-
    if (input_data[-2:] == '--') and (argu in ['-', '+']):
        return
    # 窗口已经有 ** 后面字符不能为 * 或 /
    if (input_data[-2:] == '**') and (argu in ['*', '/']):
        return

    # 输入合法将字符插入到entry窗口结尾
    entry.insert("end", argu)


# 退格(撤销输入)
def backspace(entry):
    input_len = len(entry.get())
    # 删除entry窗口中最后的字符
    entry.delete(input_len - 1)


# 清空entry内容(清空窗口)
def clear(entry):
    entry.delete(0, "end")


# 计算
def calc(entry):
    input_data = entry.get()
    # 计算前判断输入内容是否为空;首字符不能为*/;*/不能连续出现3次;
    if not input_data:
        return

    clear(entry)
    try:
        # eval() 函数用来执行一个字符串表达式，并返回表达式的值；并将执行结果转换为字符串
        output_data = str(eval(input_data))
    except Exception:
        # 将提示信息输出到窗口
        entry.insert("end", "Calculation error")
    else:
        # 将计算结果显示在窗口中
        if len(output_data) > 20:
            entry.insert("end", "Value overflow")
        else:
            entry.insert("end", output_data)


if __name__ == '__main__':
    root = tkinter.Tk()
    root.title("Yummy")

    # 框体大小可调性，分别表示x,y方向的可变性；
    root.resizable(0, 0)

    button_bg = 'pink'
    math_sign_bg = 'DarkTurquoise'
    cal_output_bg = 'Yellow'
    button_active_bg = 'gray'
    entry = tkinter.Entry(root, justify="right", font=1)
    entry.grid(row=0, column=0, columnspan=4, padx=10, pady=10)


    def place_button(text, func, func_params, bg=button_bg, **place_params):
        my_button = partial(tkinter.Button, root, bg=button_bg, padx=10, pady=3, activebackground=button_active_bg)
        button = my_button(text=text, bg=bg, command=lambda: func(*func_params))
        button.grid(**place_params)


    # 文本输入类按钮
    place_button('7', get_input, (entry, '7'), row=1, column=0, ipadx=5, pady=5)
    place_button('8', get_input, (entry, '8'), row=1, column=1, ipadx=5, pady=5)
    place_button('9', get_input, (entry, '9'), row=1, column=2, ipadx=5, pady=5)
    place_button('4', get_input, (entry, '4'), row=2, column=0, ipadx=5, pady=5)
    place_button('5', get_input, (entry, '5'), row=2, column=1, ipadx=5, pady=5)
    place_button('6', get_input, (entry, '6'), row=2, column=2, ipadx=5, pady=5)
    place_button('1', get_input, (entry, '1'), row=3, column=0, ipadx=5, pady=5)
    place_button('2', get_input, (entry, '2'), row=3, column=1, ipadx=5, pady=5)
    place_button('3', get_input, (entry, '3'), row=3, column=2, ipadx=5, pady=5)
    place_button('0', get_input, (entry, '0'), row=4, column=0, padx=8, pady=5,
                 columnspan=2, sticky=tkinter.E + tkinter.W + tkinter.N + tkinter.S)
    place_button('.', get_input, (entry, '.'), row=4, column=2, ipadx=7, padx=5, pady=5)

    # 运算输入类按钮(只是背景色不同)
    # 字符大小('+','-'宽度不一样,使用ipadx进行修正)
    place_button('+', get_input, (entry, '+'), bg=math_sign_bg, row=1, column=3, ipadx=5, pady=5)
    place_button('-', get_input, (entry, '-'), bg=math_sign_bg, row=2, column=3, ipadx=5, pady=5)
    place_button('*', get_input, (entry, '*'), bg=math_sign_bg, row=3, column=3, ipadx=5, pady=5)
    place_button('/', get_input, (entry, '/'), bg=math_sign_bg, row=4, column=3, ipadx=5, pady=5)

    # 功能输入类按钮(背景色、触发功能不同)
    place_button('<-', backspace, (entry,), row=5, column=0, ipadx=5, padx=5, pady=5)
    place_button('C', clear, (entry,), row=5, column=1, pady=5, ipadx=5)
    place_button('=', calc, (entry,), bg=cal_output_bg, row=5, column=2, ipadx=5, padx=5, pady=5,
                 columnspan=2, sticky=tkinter.E + tkinter.W + tkinter.N + tkinter.S)

    root.mainloop()


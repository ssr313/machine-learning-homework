from sympy import symbols, Eq,exp,diff
s = symbols('s')
def theta(s):
    a = 1/(1+exp(-s))
    return(a)
expr1 = theta(s)
expr2 = exp(s)/(1+exp(s))
# 使用 Eq 创建等式对象，然后使用 .simplify() 方法
eq = Eq(expr1, expr2)
simplified_eq = eq.simplify()
print(simplified_eq)
expr3 = theta(-s)
print(Eq(expr3, (1-expr2)).simplify())
expr4 = diff(theta(s),s)
expr5 = theta(s)*(1-theta(s))
print(Eq(expr4, expr5).simplify())


function G = givens(x,y)

if y == 0
    c = 1;
    s = 0;
else
    if abs(y) >= abs(x)
        cotangent = x / y;
        s = 1/sqrt(1+cotangent^2);
        c = s*cotangent;
    else
        tangent = y / x;
        c = 1/sqrt(1+tangent^2);
        s = c*tangent;
    end
end

G = [c,s;-s,c];

end

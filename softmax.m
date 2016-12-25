% Softmax operator
function b = softmax(a)
    b = exp(a);
    b = b ./ sum(b);
end
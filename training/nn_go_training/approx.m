clear all
close all
clf
lim = 80;
deg = 63;
sigmoid = '1 / (1 + exp(-x))';
silu = 'x / (1 + exp(-x))';
f = chebfun(silu,[-lim,lim],'splitting','on');
x = linspace(-lim,lim,1000);
p_mini = minimax(f, deg);
x_ext = linspace(-lim*2,lim*2,1000);
figure(1)
%y = 1/2(b-a)x + 1/2(b+a) --> x = -(b+a)/(b-a)+2y/(b-a)
plot(x,f(x), 'r'); hold on
a = -lim;
b = lim;
plot(x,p_mini(x),'b'); grid on
figure(2)
plot(x, abs(f(x)-p_mini(x))); grid on

f=poly(p_mini);
g=poly(diff(p_mini));

% in cheby form
h = chebcoeffs(p_mini);
k = chebcoeffs(diff(p_mini));

F = "var coeffsF = []float64{";
G = "var coeffsG = []float64{";
H = "var coeffsH = []float64{";
K = "var coeffsK = []float64{";
for i=1:length(f)
    F = F + sprintf("%.8e, ", f(i));
    H = H + sprintf("%.8e, ", h(i));
    if i < length(g)
        G = G + sprintf("%.8e, ", g(i));
        K = K + sprintf("%.8e, ", k(i));
    end
end

a = sprintf("var a = %d.0", -lim);
b = sprintf("var b = %d.0", lim);
F = F + "}";
G = G + "}";
H = H + "}";
K = K + "}";
fprintf("%s\n%s\n%s\n%s\n%s\n%s", a,b,F,G,H,K)
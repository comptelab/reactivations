function varargout = shaded_ci(x,y,col,trans,linewidth);
% x: x coordinates
% y: either just one y vector, or 2xN or 3xN matrix of y-data
% fstr: format ('r' or 'b--' etc)
%
% example
% x=[-10:.1:10];shaded_ci(x,[sin(x.*1.1)+1;sin(x.*1.1);sin(x*.9)-1],'r');
if nargin<3;
  col='b';
end
if nargin<4;
  trans=0.2;
end
if nargin<5;
  linewidth=2;
end

if size(y,1)==3 % also draw mean
    px=[x,fliplr(x)];
    py=[y(1,:), fliplr(y(3,:))];
    patch(px,py,1,'FaceColor',col,'EdgeColor','none');
    hold all; plot(x,y(2,:),'color',col,'LineWidth',linewidth);
end;

alpha(trans); % make patch transparent
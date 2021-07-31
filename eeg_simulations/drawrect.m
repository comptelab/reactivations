 function drawrect(x1,x2,y1,y2,style,Width,Color)

if nargin < 5;
    style = '--';
end
if nargin < 6;
    Width = 1;
end
% draws a rectangle line by line -> drawrect(0.5,1.4,9,13,'-.');
% Diego Lozano-Soldevilla 06-Sep-2013 15:18:31
line([x1 x2;x1 x2],[y1 y2],'Color','black','LineWidth',Width,'LineStyle',style,'Color',Color);
line([x1 x1;x2 x2],[y1 y1],'Color','black','LineWidth',Width,'LineStyle',style,'Color',Color);
line([x1 x1;x2 x2],[y2 y2],'Color','black','LineWidth',Width,'LineStyle',style,'Color',Color);

%% history_spline_generator
% This function uses built-in functions from the spline toolbox to
% genrerate B-spline basis functions based on the length and number of
% knots of the ongoing iteration
%%


function [Design_Matrix] = Create_splines_linspace(length, nr_knots, kill_last_spline)

history_time = 0:length; 
%knots        = logspace(log10(1), log10(length), nr_knots);   
knots        = round(linspace(1, length, nr_knots));  
knots        = unique(knots);
% A logarithmic spacing of knots means that there are more nuanced history
% effects in the immediate vicinity of an occurence of the word than at far
% off positions

knots  = augknt(knots, 4);
% This functions augments (increases) the number of node by repeating the
% outermost knots 4 times each. This is so that the B-splines at the
% extremeties still have enough knots to span. 

Design_Matrix = spcol(knots, 4, history_time);
% This is the function that actually generates the B-splines given a
% particular length and number of knots

if kill_last_spline
    Design_Matrix(:,size(Design_Matrix,2)) = [];    
end

end
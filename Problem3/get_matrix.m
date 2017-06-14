function matrix = get_matrix(filename,width,height,type)   
    
    fin = fopen(filename,'r');
    if type == 1
        I = fread(fin,width*height*3,'uint8=>uint8');
        c1 = 1;c2=1;c3=1;
        for i = 0:463202
            if mod(i,3) == 0        
                I3(c1) = I(i+1); 
                I2(c1) = I(i+2); 
                I1(c1) = I(i+3); 
                c1 = c1+1;
            end
        end
        z1 = reshape(I1,[width height]);
        z2 = reshape(I2,[width height]);
        z3 = reshape(I3,[width height]);
        % Ifinal = flipdim(imrotate(z, -90),2);
        z1=z1';z3=z3';z2=z2';
        matrix = cat(3, z3, z2, z1);
    %     imshow(z);
    else
        I = fread(fin,width*height,'uint8=>uint8');
        z1 = reshape(I,[width height]);

        % Ifinal = flipdim(imrotate(z, -90),2);
        matrix=z1';
%         figure
%         imshow(matrix);
    end
end
using Pkg
import PyPlot as py
using Images
using Plots
using DelimitedFiles
using MatrixMarket
using LinearAlgebra
using SparseArrays


enable_mul = false
enable_cg = true
enable_bicrec = false
enable_esolv = false


function getTikhonovMatrixSparse(img,lx,ly; homog=0, mu = 30, err =0.0)
    n,m = size(img)    
    Wx = zeros(n,m)
    Wy = zeros(n,m)
    
    function imf(x,y)
        if x>n || x<1 || y>m || y<1
            return 0 
        end
        return img[x,y]
    end
    
    for x=1:n, y=1:m
        Wx[x,y] = lx*(1- abs(imf(x,y)-imf(x-1,y)))^mu + err
        Wy[x,y] = ly*(1- abs(imf(x,y)-imf(x,y-1)))^mu + err
        if homog==1
            Wx[x,y] = lx
            Wy[x,y] = ly
        end
    end
    Wx[1,:] .= err
    Wy[:,1] .= err

    function wx(x,y)
        if x>n || x<1 || y>m || y<1
            return 0 
        end
        return Wx[x,y]
    end
    function wy(x,y)
        if x>n || x<1 || y>m || y<1
            return 0 
        end
        return Wy[x,y]
    end
    
    function st(A,val,i,j,x,y)
        if i>n*m || i<1 || j> n*m || j<1
            return 
        end
        if x>n || x<1 || y>m || y<1
            return 
        end
        A[i,j] = val
        return
    end
    
    I = zeros(5*n*m)
    J = zeros(5*n*m)
    V = zeros(5*n*m)
    
    #fill matrix
    
    nrow = 1
    for x = 1:n, y=1:m
        i = (y-1)*n + x
        I[nrow] = i
        J[nrow] = i
        V[nrow] = 1 + wx(x,y) + wx(x+1,y) + wy(x,y) + wy(x,y+1)
        nrow +=1
    end
    
      
    for x=1:n, y=1:m
        i = (y-1)*n + x
        j = i + m
        if !(i>n*m || i<1 || j> n*m || j<1)
            if !(x>n || x<1 || y+1>m || y+1<1)
                I[nrow] = i
                J[nrow] = j
                V[nrow] = -wy(x,y+1)
                nrow +=1
            end
        end
    end
    
      
    for x=1:n, y=1:m
        i = (y-1)*n + x
        j = i - m
        if !(i>n*m || i<1 || j> n*m || j<1)
            if !(x>n || x<1 || y-1>m || y-1<1)
                I[nrow] = i
                J[nrow] = j
                V[nrow] = -wy(x,y)
                nrow +=1
            end
        end
    end
    
    for x=1:n, y=1:m
        i = (y-1)*n + x
        j = i + 1
        if !(i>n*m || i<1 || j> n*m || j<1)
            if !(x+1>n || x+1<1 || y>m || y<1)
                I[nrow] = i
                J[nrow] = j
                V[nrow] = -wx(x+1,y)
                nrow +=1
            end
        end
    end
    
    for x=1:n, y=1:m
        i = (y-1)*n + x
        j = i - 1
        if !(i>n*m || i<1 || j> n*m || j<1)
            if !(x-1>n || x-1<1 || y>m || y<1)
                I[nrow] = i
                J[nrow] = j
                V[nrow] = -wx(x,y)
                nrow +=1
            end
        end
    end

   
    
    I = I[1:nrow-1]
    J = J[1:nrow-1]
    V = V[1:nrow-1]
    
    A = sparse(I,J,V)
    
    return Wx, Wy, A
end


function writeArraymm(img)
    open("b.mm", "w") do io
        write(io, "%%MatrixMarket matrix array real general\n")
        write(io, string(length(img)) * " 1\n")
        for i = 1:length(img)
            write(io, string(vec(img)[i])*"\n")
        end

        end;
end;

function chooseImg(path,S,lx,ly,mu, nit_esolv)
    img_path = path
    #load image
    path = "../data/" * path
    img = load(path)
    img = imresize(img, (S,S))
    img = img |>channelview .|>Float64
    img = img[1,:,:];

    #solve exact problem
    Wx, Wy, A = getTikhonovMatrixSparse(img,lx,ly, homog = 0, mu = mu, err =0.0);
    esolv = reshape(cholesky(A)\vec(img),size(img))

    if enable_esolv
        avg_dt = 0

        for i=1:nit_esolv
            dt = @timed begin
                esolv = reshape(cholesky(A)\vec(img),size(img)) #repeat to avoid compilation time
            end
            dt = dt.time #avoids counting compile time
            avg_dt += dt
        end
        avg_dt /= nit_esolv
    
        filename =  split(img_path,'.')[1]
        writedlm("../edgeaware_times/$filename-chol.csv", avg_dt, ",")
    end
    #write input and exact
    write("input.bin",collect(vec(img')));
    write("esolv.bin",collect(vec(esolv')));

    if enable_mul
        MatrixMarket.mmwrite("A.mm", A) #write inputs for multigrid solver
        writeArraymm(img)
    end
    return A,img,esolv
end;

function iterateBICRecursive(S, num_it)
    errs_rec = []
    dts_rec = []

    io = IOBuffer();
    command = `./recursiveTikhonov $S $num_it 1`;
    run(pipeline(command, stdout=io))
    log = String(take!(io))
    print(log)
    values = readdlm("logfile.txt",',')
    dts_rec = values[:,1]/1000; #measure in seconds
    errs_rec = values[:,2]; 
    return dts_rec, errs_rec
end;


function iterateCG(S, num_it)
    errs_cg = []
    dts_cg = []

    io = IOBuffer();
    command = `./recursiveTikhonov $S $num_it 0`;
    run(pipeline(command, stdout=io))
    log = String(take!(io))
    print(log)
    values = readdlm("logfile.txt",',')
    dts_cg = values[:,1]/1000; #measure in seconds
    errs_cg = values[:,2]; 
    return dts_cg, errs_cg
end;


function iterateMultigrid(S, nit, esolv)
    errs_mul = []
    dts_mul = []
    str = []
    
    for it = 1:2:nit #TODO:
        io = IOBuffer();
        command = ` env OMP_NUM_THREADS=1 ./amgcl_solve x.mm A.mm b.mm cg smoothed_aggregation spai0 $it`
        run(pipeline(command, stdout=io))
        str = String(take!(io))
        
        #SOLVE + SETUP TIME
        myregex = r"(.*(?:\[  solve).*)" 
        m = match(myregex, str)
        values = split(filter(x->('0'<=x<='9')||(x==']')||(x=='.'),m.captures[1]),']')
        dt = parse(Float64,values[1]) #measured in seconds
        
        myregex = r"(.*(?:\[  setup).*)" 
        m = match(myregex, str)
        values = split(filter(x->('0'<=x<='9')||(x==']')||(x=='.'),m.captures[1]),']')
        dt += parse(Float64,values[1]); #measured in seconds
        
        #COMPUTE RELATIVE L2 ERROR
        aux = readdlm("x.mm", ',');
        approx = reshape(aux[13:end],(S,S));
        err = norm(approx .- esolv)/norm(esolv)

        append!(dts_mul, dt)
        append!(errs_mul, err)
    end

    return dts_mul, errs_mul
end;

function averageIterate(f, n_avg)   
    a0,b0 = f()
    a = a0
    b = b0
    for it = 1:n_avg
        a,b = f()
        a0 += a
        b0 += b
        
    end
    a0 = a0 ./ n_avg
    b0 = b0 ./ n_avg
    
    return a0,b0
end;

function writeExecution(dt, err, filename)
    #write time,err
    mat = zeros((length(dt),2))
    mat[:,1] = dt
    mat[:,2] = err;
    writedlm("../edgeaware_times/$filename.csv", mat, ",")
end;




function main()

    #INITIALIZE --------------------------------------------------------------------------
    #filter parameters
    p = readdlm("input_params.txt",',')
    lx = p[1]
    ly = p[2]
    mu = p[3]
    S =  Int(p[4]) #image dimensions SxS
    filepaths = readdir("../data");
    BLAS.set_num_threads(1) #avoid multithreading

    n_avg = 10; #number of times iterations are averaged for execution time approximation
    num_it_rec = 200
    num_it_mul = 20
    num_it_cg = 200;
    min_errs_multigrid= LinRange(0.04, 1e-6, num_it_mul)

       

    #ITERATE --------------------------------------------------------------------------
    img_idx = 1
    for img_idx = 1:length(filepaths)
        img_path = filepaths[img_idx]
        print("Input $img_path\n")
        filename =  split(img_path,'.')[1]
        A,img, esolv = chooseImg(img_path,S,lx,ly,mu, n_avg);

        #define lambda functions
        f_cg() = iterateCG(S,num_it_cg);
        f_bicrec() = iterateBICRecursive(S, num_it_rec)
        
        
        if enable_bicrec
            #execute solvers
            print("Solving Recursive\n")
            dts_rec, errs_rec = averageIterate(f_rec, n_avg);
            writeExecution(dts_rec, errs_rec, filename * "-bicrec")
        end

        if enable_cg
            print("Solving CG C++\n")
            dts_cg, errs_cg = averageIterate(f_cg, n_avg);
            writeExecution(dts_cg, errs_cg, filename * "-cg")
        end
        
        if enable_mul
            f_mul() = iterateMultigrid(S, num_it_mul, esolv);
            print("Solving Multigrid\n")
            dts_mul, errs_mul = averageIterate(f_mul, n_avg);
            writeExecution(dts_mul, errs_mul, filename * "-mul")
        end
    end
        
end

main()
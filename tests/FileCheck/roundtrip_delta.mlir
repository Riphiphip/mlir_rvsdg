rvsdg.omegaNode():{ 
    %10 = rvsdg.deltaNode(): 
        (): {
        %11 = arith.constant 10:i16
        rvsdg.deltaResult(%11:i16)
    }->!llvm.ptr<i16>

    %11 = rvsdg.deltaNode(): 
        (): {
        %11 = arith.constant 10:i16
        rvsdg.deltaResult(%11:i16)
    }->!rvsdg.ptr<i16>
    rvsdg.omegaResult(%10:!llvm.ptr<i16>, %11:!rvsdg.ptr<i16>)
}
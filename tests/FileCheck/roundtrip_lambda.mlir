rvsdg.omegaNode (): {
    %ctx0 = arith.constant 4: i32
    %ctx1 = arith.constant 5.0: f32

    %l = rvsdg.lambdaNode <(i32)->(f32)> (%ctx0:i32, %ctx1:f32):
        (%arg: i32, %ctx0: i32, %ctx1:f32): {
        %0 = arith.muli %ctx0, %arg: i32
        %1 = arith.sitofp %0: i32 to f32
        %2 = arith.addf %1, %ctx1: f32
        rvsdg.lambdaResult(%2:f32)
    }

    %param = arith.constant 100: i32
    %res= rvsdg.applyNode %l:<(i32)->(f32)>(%param:i32) -> f32
    rvsdg.omegaResult()
}
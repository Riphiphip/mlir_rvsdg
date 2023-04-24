


%lambda1, %lambda2 = rvsdg.phiNode (): 
(%lambda1_ref:!rvsdg.lambdaRef<(f32)->(f32)>, %lambda2_ref:!rvsdg.lambdaRef<()->(i32)>):{
    %lambda1 = rvsdg.lambdaNode <(f32)->(f32)> ():
        (%test:f32): {
        %0 = arith.constant 1.0: f32
        rvsdg.lambdaResult(%0:f32)
    }
    %lambda2 = rvsdg.lambdaNode <()->(i32)> ():
        (): {
        %0 = arith.constant 1: i32
        rvsdg.lambdaResult(%0:i32)
    }
    rvsdg.phiResult(%lambda1:!rvsdg.lambdaRef<(f32)->(f32)>, %lambda2:!rvsdg.lambdaRef<()->(i32)>)
} -> !rvsdg.lambdaRef<(f32)->(f32)>, !rvsdg.lambdaRef<()->(i32)>


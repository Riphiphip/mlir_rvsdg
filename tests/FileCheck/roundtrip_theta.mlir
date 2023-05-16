rvsdg.omegaNode(): {
    %test0 = arith.constant 1.0: f32
    %test1 = arith.constant 1.0: f64

    %res0:2 = rvsdg.thetaNode (%test0:f32, %test1:f64): 
    (%0: f32, %1: f64): {
        %test = arith.constant 1.0: f32
        %predicate = rvsdg.constantCtrl 0: <2>
        rvsdg.thetaResult(%predicate) : (%0:f32, %1:f64)
    } -> f32, f64

    rvsdg.omegaResult()
}

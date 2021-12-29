function getDVs(x, y, targetX, targetY, tof) {
    let r1 = [[x], [y], [0]]
    let r2 = [[targetX], [targetY], [0]]
    let phi = phiMatrix(tof);
    let v1 = math.multiply(math.inv(phi.rv), math.subtract(r2, math.multiply(phi.rr, r1)));
    // let v2 = add(multiply(phi.vr, r1), multiply(phi.vv, v1));
    return [v1[0][0], v1[1][0]];
}


function phiMatrix(t = 0, n = 2 * Math.PI / 86164) {
    let nt = n * t;
    let cnt = Math.cos(nt);
    let snt = Math.sin(nt);
    return {
        rr: [
            [4 - 3 * cnt, 0, 0],
            [6 * (snt - nt), 1, 0],
            [0, 0, cnt]
        ],
        rv: [
            [snt / n, 2 * (1 - cnt) / n, 0],
            [2 * (cnt - 1) / n, (4 * snt - 3 * nt) / n, 0],
            [0, 0, snt / n]
        ],
        vr: [
            [3 * n * snt, 0, 0],
            [6 * n * (cnt - 1), 0, 0],
            [0, 0, -n * snt]
        ],
        vv: [
            [cnt, 2 * snt, 0],
            [-2 * snt, 4 * cnt - 3, 0],
            [0, 0, cnt]
        ]
    };
}


const target = [0,0]

const limits = {
    r: [-50, 50],
    i: [-50, 50],
    tof: [0.5, 6]
}

let data = []
for (let ii = 0; ii < 10000; ii++) {

    let r = limits.r[0] + Math.random() * (limits.r[1] - limits.r[0])
    let i = limits.i[0] + Math.random() * (limits.i[1] - limits.i[0])
    let tof = limits.tof[0] + Math.random() * (limits.tof[1] - limits.tof[0])
    let vs = getDVs(r,i, target[0], target[1], tof * 3600)
    data.push({x: r, y: i, tof, dX: vs[0], dY: vs[1]})
    
}
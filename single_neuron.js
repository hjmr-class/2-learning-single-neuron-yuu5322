let teach_num = 4;
let inp_num = 2;

let alpha = 0.01;

let w = [];
let dw = [];
let theta, d_theta

let teach_x = [[0, 0], [0, 1], [1, 0], [1, 1]];
let teach_y = [0, 1, 0, 1];

function rand_one(){
    let r =  Math.random();
    return r;
}

function sigmoid(x){
    return 1.0 / (1.0 + Math.exp(-x));
}

function forward(x){
    let i = 0;
    let u = 0;
    for (let i = 0; i < inp_num; i++){
        u += w[i] * x[i];
    }
    u += theta;
    return sigmoid(u);
}

function func_error(){
    let t = 0;
    let e = 0;
    for(let t = 0; t < teach_num; t++){
    let y = forward(teach_x[t]);
    e += 0.5 * (y - teach_y[t] * (y - teach_y[t]));
    }
    return e;
}

function clear_dw(){
    let i = 0;
    for(let i = 0; i < inp_num; i++){
    dw[i] = 0;
    }
    d_theta = 0
}

function calc_dw(){
    let t = 0;
    let i = 0;
    for(t = 0; t < teach_num; t++){
    let x_t = teach_x[t];
    let y = forward(x_t);
    y_hat = teach_y[t];
    for(i =0; i < inp_num; i++){
      dw[i] += (y - y_hat) * y * (1 - y) * x_t[i];
    }
    d_theta += (y - y_hat) * y * (1 - y);
    }
}

function init_w(){
    let i = 0;
    for(let i = 0; i < inp_num + 1; i++){
    w[i] = rand_one() * 2 -1.0;
    }
  theta = rand_one() * 2 - 1.0;
}

function update_w(){
    let i = 0;
    for(i = 0; i < inp_num; i++){
    w[i] -= alpha * dw[i];
    }
  theta -= alpha * d_theta
}

function main(){
    let t = 0;
    let loop = 0;

    init_w();

    for(let t = 0; t < teach_num; t++){
        let y = forward(teach_x[t]);
        console.log(t + ": y =" + y + "<--> y_hat =" + teach_y[t]);
    }
    for (loop = 0; loop < 100000; loop++) {
        if(loop % 1000 == 0){
            console.log(loop, func_error());
        }
        clear_dw()
        calc_dw()
        update_w()
        }

    console.log(loop, func_error())

    for(let t = 0; t < teach_num; t++){
    let y = forward(teach_x[t]);
    console.log(t +": " +  "y=" + y + "<---> y_hat =" + teach_y[t]);
    }
}

main();
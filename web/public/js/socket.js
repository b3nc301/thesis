

function getData(){
var host = 'ws://'+ip+':'+port;
var socket = new WebSocket(host);
var data = "status;0";
socket.addEventListener('open', function (event) {
    socket.send(data);
});
socket.addEventListener('message', function (event) {
    console.log( event.data);
    var data = JSON.parse(event.data);
    document.getElementById("src").setAttribute("value",data.src);
    document.getElementById("frames").setAttribute("value",data.frames);
    document.getElementById("conf").setAttribute("value",data.conf);
    document.getElementById("max").setAttribute("value",data.max);
    document.getElementById("min").setAttribute("value",data.min);
});
}

function startDetector(){
    var host = 'ws://'+ip+':'+port;
    var socket = new WebSocket(host);
    var data = 'start;{"src":"'+document.getElementById("src").value+
    '", "frames":"'+document.getElementById("frames").value+
    '", "conf":"'+document.getElementById("conf").value+
    '", "max":"'+document.getElementById("max").value+
    '", "min":"'+document.getElementById("min").value+
    '"}';
    var alert = document.getElementById('liveAlert')

    socket.addEventListener('open', function (event) {
        socket.send(data);
    });
    socket.addEventListener('message', function (event) {
        var recdata = JSON.parse(event.data);
        console.log(recdata);

        if(recdata.status == "ok"){
            var wrapper = document.createElement('div')
            wrapper.innerHTML = '<div class="alert alert-success alert-dismissible" role="alert">Detector has been started succesfuly.<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>'
            alert.append(wrapper)
        }
        else {
            var wrapper = document.createElement('div')
            wrapper.innerHTML = '<div class="alert alert-danger alert-dismissible" role="alert">An error has been occured.<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>'
            alert.append(wrapper)
        }
    });
}

function stopDetector(){
    var host = 'ws://'+ip+':'+port;
    var socket = new WebSocket(host);
    var data = "stop;0";
    var alert = document.getElementById('liveAlert')
    socket.addEventListener('open', function (event) {
        socket.send(data);
    });
    socket.addEventListener('message', function (event) {
        console.log( event.data);
        var recdata = JSON.parse(event.data);
        if(recdata.status == "ok"){
            var wrapper = document.createElement('div')
            wrapper.innerHTML = '<div class="alert alert-success alert-dismissible" role="alert">Detector has been stopped succesfuly.<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>'
            alert.append(wrapper)
        }
        else {
            var wrapper = document.createElement('div')
            wrapper.innerHTML = '<div class="alert alert-danger alert-dismissible" role="alert">An error has been occured.<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>'
            alert.append(wrapper)
        }
    });
    }
    function changeData(){
        var host = 'ws://'+ip+':'+port;
        var socket = new WebSocket(host);
        var data='change;{"src":"'+document.getElementById("src").value+
        '", "frames":"'+document.getElementById("frames").value+
        '", "conf":"'+document.getElementById("conf").value+
        '", "max":"'+document.getElementById("max").value+
        '", "min":"'+document.getElementById("min").value+
        '"}';
        var alert = document.getElementById('liveAlert')
        socket.addEventListener('open', function (event) {
            socket.send(data);
        });
        socket.addEventListener('message', function (event) {
            console.log( event.data);
            var recdata = JSON.parse(event.data);
            if(recdata.status == "ok"){
                var wrapper = document.createElement('div')
                wrapper.innerHTML = '<div class="alert alert-success alert-dismissible" role="alert">Detector has been changed succesfuly.<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>'
                alert.append(wrapper)
            }
            else {
                var wrapper = document.createElement('div')
                wrapper.innerHTML = '<div class="alert alert-danger alert-dismissible" role="alert">An error has been occured.<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>'
                alert.append(wrapper)
            }
        });
        }

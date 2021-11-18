<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class Socket extends Controller
{
    public static function getData(){
        $socket = socket_create(AF_INET, SOCK_STREAM, SOL_TCP);
        $result = socket_connect($socket, env('DETECTOR_IP'), env('DETECTOR_PORT'));
        if ($result === false) {
            echo "socket_connect() failed.\nReason: ($result) " . socket_strerror(socket_last_error($socket)) . "\n";
        } else {
            echo "OK.\n";
        }

        $msg='{"status","0"}';
        echo "Sending HTTP HEAD request...";
        socket_write($socket, $msg, strlen($msg));
        echo "OK.\n";
        echo "Reading response:\n\n";
        $return = "";
        while ($out = socket_read($socket, 2048)) {
            $return .= $out;
        }
        echo "Closing socket...";
        socket_close($socket);
        echo "OK.\n\n";
        return $return;

    }
}

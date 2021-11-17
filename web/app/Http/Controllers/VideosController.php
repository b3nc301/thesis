<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Videos;
use Exception;
use Illuminate\Support\Facades\Auth;



class VideosController extends Controller
{
    public static function getVideos(Request $request){
        if($request->filled('starttime')){
            $starttime = $request->input('starttime');
        }
        else{
            $starttime =  date("Y-m-d H:i:s", strtotime("-1 hours"));
        }
        if($request->filled('endtime')){
            $endtime = $request->input('endtime');
        }
        else{
            $endtime =  date("Y-m-d H:i:s");
        }
        return view('vids', ['videos' => Videos::whereBetween('videoDate', [$starttime, $endtime])->cursor()]);
    }
    public static function deleteVideo(Request $request){
        if(Auth::check()){


           return unlink(public_path() ."/". Videos::find($request->input('id'))->videoURL) && Videos::find($request->input('id'))->delete();
        }
    }
}

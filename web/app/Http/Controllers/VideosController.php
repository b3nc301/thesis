<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Videos;
use Exception;
use Illuminate\Support\Facades\Auth;



class VideosController extends Controller
{
    public static function getAllVideos(){
        return Videos::all();
    }
    public static function deleteVideo(Request $request){
        if(Auth::check()){


           return unlink(public_path() ."/". Videos::find($request->input('id'))->videoURL) && Videos::find($request->input('id'))->delete();
        }
    }
}

<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Events;
use Illuminate\Support\Facades\Auth;


class EventsController extends Controller
{
    public static function getEvents(Request $request){
        date_default_timezone_set("Europe/Budapest");
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
        if($request->filled('level')){
            return view('events', ['events' => Events::whereBetween('time', [$starttime, $endtime])->where('level',$request->input('level'))->cursor()]);
        }
        else{
        return view('events', ['events' => Events::whereBetween('time', [$starttime, $endtime])->cursor()]);
        }
    }
    public static function deleteEvent(Request $request){
        if(Auth::check()){
            Events::find($request->input('id'))->delete();
        }
    }


}

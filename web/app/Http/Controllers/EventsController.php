<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Events;
use Illuminate\Support\Facades\Auth;


class EventsController extends Controller
{
    public static function getAllEvents(){
        return Events::all();
    }
    public static function deleteEvent(Request $request){
        if(Auth::check()){
            Events::find($request->input('id'))->delete();
        }
    }
}

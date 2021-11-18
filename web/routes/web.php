<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\EventsController;
use App\Http\Controllers\VideosController;
use App\Http\Controllers\Socket;


/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| contains the "web" middleware group. Now create something great!
|
*/

Route::get('/', function () {
    return view('dashboard');
})->middleware(['auth'])->name('dashboard');

Route::get('/login', function () {
    return view('dashboard');
})->middleware(['auth'])->name('dashboard');


Route::get('/live', function () {
    return view('live');
})->middleware(['auth'])->name('live');

Route::get('/events',[EventsController::class, 'getEvents'])->middleware(['auth'])->name('events');

Route::get('/vids', [VideosController::class, 'getVideos'])->middleware(['auth'])->name('vids');

Route::post('/deleteVideo', [VideosController::class, 'deleteVideo']);



require __DIR__.'/auth.php';

<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Events extends Model
{
    use HasFactory;
    protected $table = 'events';
    protected $primaryKey = 'id';
    public $timestamps = false;

    protected $fillable = [
		'xbox',
		'ybox',
		'wbox',
		'hbox',
        'classID',
        'time',
        'videoTime',
        'videoID',
        'level'
	];
    protected $casts = [
        'time' => 'datetime'
    ];
    public function video()
    {
        return $this->belongsTo(Videos::class, 'videoID');
    }
    public function class()
    {
        return $this->belongsTo(Classes::class, 'classID');
    }


}

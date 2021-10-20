(function (global, factory) {
	typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
	typeof define === 'function' && define.amd ? define(['exports'], factory) :
	(global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.splink_vis_utils = {}));
})(this, (function (exports) { 'use strict';

	var commonjsGlobal = typeof globalThis !== 'undefined' ? globalThis : typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : typeof self !== 'undefined' ? self : {};

	var lodash_clonedeep = {exports: {}};

	/**
	 * lodash (Custom Build) <https://lodash.com/>
	 * Build: `lodash modularize exports="npm" -o ./`
	 * Copyright jQuery Foundation and other contributors <https://jquery.org/>
	 * Released under MIT license <https://lodash.com/license>
	 * Based on Underscore.js 1.8.3 <http://underscorejs.org/LICENSE>
	 * Copyright Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
	 */

	(function (module, exports) {
	/** Used as the size to enable large array optimizations. */
	var LARGE_ARRAY_SIZE = 200;

	/** Used to stand-in for `undefined` hash values. */
	var HASH_UNDEFINED = '__lodash_hash_undefined__';

	/** Used as references for various `Number` constants. */
	var MAX_SAFE_INTEGER = 9007199254740991;

	/** `Object#toString` result references. */
	var argsTag = '[object Arguments]',
	    arrayTag = '[object Array]',
	    boolTag = '[object Boolean]',
	    dateTag = '[object Date]',
	    errorTag = '[object Error]',
	    funcTag = '[object Function]',
	    genTag = '[object GeneratorFunction]',
	    mapTag = '[object Map]',
	    numberTag = '[object Number]',
	    objectTag = '[object Object]',
	    promiseTag = '[object Promise]',
	    regexpTag = '[object RegExp]',
	    setTag = '[object Set]',
	    stringTag = '[object String]',
	    symbolTag = '[object Symbol]',
	    weakMapTag = '[object WeakMap]';

	var arrayBufferTag = '[object ArrayBuffer]',
	    dataViewTag = '[object DataView]',
	    float32Tag = '[object Float32Array]',
	    float64Tag = '[object Float64Array]',
	    int8Tag = '[object Int8Array]',
	    int16Tag = '[object Int16Array]',
	    int32Tag = '[object Int32Array]',
	    uint8Tag = '[object Uint8Array]',
	    uint8ClampedTag = '[object Uint8ClampedArray]',
	    uint16Tag = '[object Uint16Array]',
	    uint32Tag = '[object Uint32Array]';

	/**
	 * Used to match `RegExp`
	 * [syntax characters](http://ecma-international.org/ecma-262/7.0/#sec-patterns).
	 */
	var reRegExpChar = /[\\^$.*+?()[\]{}|]/g;

	/** Used to match `RegExp` flags from their coerced string values. */
	var reFlags = /\w*$/;

	/** Used to detect host constructors (Safari). */
	var reIsHostCtor = /^\[object .+?Constructor\]$/;

	/** Used to detect unsigned integer values. */
	var reIsUint = /^(?:0|[1-9]\d*)$/;

	/** Used to identify `toStringTag` values supported by `_.clone`. */
	var cloneableTags = {};
	cloneableTags[argsTag] = cloneableTags[arrayTag] =
	cloneableTags[arrayBufferTag] = cloneableTags[dataViewTag] =
	cloneableTags[boolTag] = cloneableTags[dateTag] =
	cloneableTags[float32Tag] = cloneableTags[float64Tag] =
	cloneableTags[int8Tag] = cloneableTags[int16Tag] =
	cloneableTags[int32Tag] = cloneableTags[mapTag] =
	cloneableTags[numberTag] = cloneableTags[objectTag] =
	cloneableTags[regexpTag] = cloneableTags[setTag] =
	cloneableTags[stringTag] = cloneableTags[symbolTag] =
	cloneableTags[uint8Tag] = cloneableTags[uint8ClampedTag] =
	cloneableTags[uint16Tag] = cloneableTags[uint32Tag] = true;
	cloneableTags[errorTag] = cloneableTags[funcTag] =
	cloneableTags[weakMapTag] = false;

	/** Detect free variable `global` from Node.js. */
	var freeGlobal = typeof commonjsGlobal == 'object' && commonjsGlobal && commonjsGlobal.Object === Object && commonjsGlobal;

	/** Detect free variable `self`. */
	var freeSelf = typeof self == 'object' && self && self.Object === Object && self;

	/** Used as a reference to the global object. */
	var root = freeGlobal || freeSelf || Function('return this')();

	/** Detect free variable `exports`. */
	var freeExports = exports && !exports.nodeType && exports;

	/** Detect free variable `module`. */
	var freeModule = freeExports && 'object' == 'object' && module && !module.nodeType && module;

	/** Detect the popular CommonJS extension `module.exports`. */
	var moduleExports = freeModule && freeModule.exports === freeExports;

	/**
	 * Adds the key-value `pair` to `map`.
	 *
	 * @private
	 * @param {Object} map The map to modify.
	 * @param {Array} pair The key-value pair to add.
	 * @returns {Object} Returns `map`.
	 */
	function addMapEntry(map, pair) {
	  // Don't return `map.set` because it's not chainable in IE 11.
	  map.set(pair[0], pair[1]);
	  return map;
	}

	/**
	 * Adds `value` to `set`.
	 *
	 * @private
	 * @param {Object} set The set to modify.
	 * @param {*} value The value to add.
	 * @returns {Object} Returns `set`.
	 */
	function addSetEntry(set, value) {
	  // Don't return `set.add` because it's not chainable in IE 11.
	  set.add(value);
	  return set;
	}

	/**
	 * A specialized version of `_.forEach` for arrays without support for
	 * iteratee shorthands.
	 *
	 * @private
	 * @param {Array} [array] The array to iterate over.
	 * @param {Function} iteratee The function invoked per iteration.
	 * @returns {Array} Returns `array`.
	 */
	function arrayEach(array, iteratee) {
	  var index = -1,
	      length = array ? array.length : 0;

	  while (++index < length) {
	    if (iteratee(array[index], index, array) === false) {
	      break;
	    }
	  }
	  return array;
	}

	/**
	 * Appends the elements of `values` to `array`.
	 *
	 * @private
	 * @param {Array} array The array to modify.
	 * @param {Array} values The values to append.
	 * @returns {Array} Returns `array`.
	 */
	function arrayPush(array, values) {
	  var index = -1,
	      length = values.length,
	      offset = array.length;

	  while (++index < length) {
	    array[offset + index] = values[index];
	  }
	  return array;
	}

	/**
	 * A specialized version of `_.reduce` for arrays without support for
	 * iteratee shorthands.
	 *
	 * @private
	 * @param {Array} [array] The array to iterate over.
	 * @param {Function} iteratee The function invoked per iteration.
	 * @param {*} [accumulator] The initial value.
	 * @param {boolean} [initAccum] Specify using the first element of `array` as
	 *  the initial value.
	 * @returns {*} Returns the accumulated value.
	 */
	function arrayReduce(array, iteratee, accumulator, initAccum) {
	  var index = -1,
	      length = array ? array.length : 0;

	  if (initAccum && length) {
	    accumulator = array[++index];
	  }
	  while (++index < length) {
	    accumulator = iteratee(accumulator, array[index], index, array);
	  }
	  return accumulator;
	}

	/**
	 * The base implementation of `_.times` without support for iteratee shorthands
	 * or max array length checks.
	 *
	 * @private
	 * @param {number} n The number of times to invoke `iteratee`.
	 * @param {Function} iteratee The function invoked per iteration.
	 * @returns {Array} Returns the array of results.
	 */
	function baseTimes(n, iteratee) {
	  var index = -1,
	      result = Array(n);

	  while (++index < n) {
	    result[index] = iteratee(index);
	  }
	  return result;
	}

	/**
	 * Gets the value at `key` of `object`.
	 *
	 * @private
	 * @param {Object} [object] The object to query.
	 * @param {string} key The key of the property to get.
	 * @returns {*} Returns the property value.
	 */
	function getValue(object, key) {
	  return object == null ? undefined : object[key];
	}

	/**
	 * Checks if `value` is a host object in IE < 9.
	 *
	 * @private
	 * @param {*} value The value to check.
	 * @returns {boolean} Returns `true` if `value` is a host object, else `false`.
	 */
	function isHostObject(value) {
	  // Many host objects are `Object` objects that can coerce to strings
	  // despite having improperly defined `toString` methods.
	  var result = false;
	  if (value != null && typeof value.toString != 'function') {
	    try {
	      result = !!(value + '');
	    } catch (e) {}
	  }
	  return result;
	}

	/**
	 * Converts `map` to its key-value pairs.
	 *
	 * @private
	 * @param {Object} map The map to convert.
	 * @returns {Array} Returns the key-value pairs.
	 */
	function mapToArray(map) {
	  var index = -1,
	      result = Array(map.size);

	  map.forEach(function(value, key) {
	    result[++index] = [key, value];
	  });
	  return result;
	}

	/**
	 * Creates a unary function that invokes `func` with its argument transformed.
	 *
	 * @private
	 * @param {Function} func The function to wrap.
	 * @param {Function} transform The argument transform.
	 * @returns {Function} Returns the new function.
	 */
	function overArg(func, transform) {
	  return function(arg) {
	    return func(transform(arg));
	  };
	}

	/**
	 * Converts `set` to an array of its values.
	 *
	 * @private
	 * @param {Object} set The set to convert.
	 * @returns {Array} Returns the values.
	 */
	function setToArray(set) {
	  var index = -1,
	      result = Array(set.size);

	  set.forEach(function(value) {
	    result[++index] = value;
	  });
	  return result;
	}

	/** Used for built-in method references. */
	var arrayProto = Array.prototype,
	    funcProto = Function.prototype,
	    objectProto = Object.prototype;

	/** Used to detect overreaching core-js shims. */
	var coreJsData = root['__core-js_shared__'];

	/** Used to detect methods masquerading as native. */
	var maskSrcKey = (function() {
	  var uid = /[^.]+$/.exec(coreJsData && coreJsData.keys && coreJsData.keys.IE_PROTO || '');
	  return uid ? ('Symbol(src)_1.' + uid) : '';
	}());

	/** Used to resolve the decompiled source of functions. */
	var funcToString = funcProto.toString;

	/** Used to check objects for own properties. */
	var hasOwnProperty = objectProto.hasOwnProperty;

	/**
	 * Used to resolve the
	 * [`toStringTag`](http://ecma-international.org/ecma-262/7.0/#sec-object.prototype.tostring)
	 * of values.
	 */
	var objectToString = objectProto.toString;

	/** Used to detect if a method is native. */
	var reIsNative = RegExp('^' +
	  funcToString.call(hasOwnProperty).replace(reRegExpChar, '\\$&')
	  .replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g, '$1.*?') + '$'
	);

	/** Built-in value references. */
	var Buffer = moduleExports ? root.Buffer : undefined,
	    Symbol = root.Symbol,
	    Uint8Array = root.Uint8Array,
	    getPrototype = overArg(Object.getPrototypeOf, Object),
	    objectCreate = Object.create,
	    propertyIsEnumerable = objectProto.propertyIsEnumerable,
	    splice = arrayProto.splice;

	/* Built-in method references for those with the same name as other `lodash` methods. */
	var nativeGetSymbols = Object.getOwnPropertySymbols,
	    nativeIsBuffer = Buffer ? Buffer.isBuffer : undefined,
	    nativeKeys = overArg(Object.keys, Object);

	/* Built-in method references that are verified to be native. */
	var DataView = getNative(root, 'DataView'),
	    Map = getNative(root, 'Map'),
	    Promise = getNative(root, 'Promise'),
	    Set = getNative(root, 'Set'),
	    WeakMap = getNative(root, 'WeakMap'),
	    nativeCreate = getNative(Object, 'create');

	/** Used to detect maps, sets, and weakmaps. */
	var dataViewCtorString = toSource(DataView),
	    mapCtorString = toSource(Map),
	    promiseCtorString = toSource(Promise),
	    setCtorString = toSource(Set),
	    weakMapCtorString = toSource(WeakMap);

	/** Used to convert symbols to primitives and strings. */
	var symbolProto = Symbol ? Symbol.prototype : undefined,
	    symbolValueOf = symbolProto ? symbolProto.valueOf : undefined;

	/**
	 * Creates a hash object.
	 *
	 * @private
	 * @constructor
	 * @param {Array} [entries] The key-value pairs to cache.
	 */
	function Hash(entries) {
	  var index = -1,
	      length = entries ? entries.length : 0;

	  this.clear();
	  while (++index < length) {
	    var entry = entries[index];
	    this.set(entry[0], entry[1]);
	  }
	}

	/**
	 * Removes all key-value entries from the hash.
	 *
	 * @private
	 * @name clear
	 * @memberOf Hash
	 */
	function hashClear() {
	  this.__data__ = nativeCreate ? nativeCreate(null) : {};
	}

	/**
	 * Removes `key` and its value from the hash.
	 *
	 * @private
	 * @name delete
	 * @memberOf Hash
	 * @param {Object} hash The hash to modify.
	 * @param {string} key The key of the value to remove.
	 * @returns {boolean} Returns `true` if the entry was removed, else `false`.
	 */
	function hashDelete(key) {
	  return this.has(key) && delete this.__data__[key];
	}

	/**
	 * Gets the hash value for `key`.
	 *
	 * @private
	 * @name get
	 * @memberOf Hash
	 * @param {string} key The key of the value to get.
	 * @returns {*} Returns the entry value.
	 */
	function hashGet(key) {
	  var data = this.__data__;
	  if (nativeCreate) {
	    var result = data[key];
	    return result === HASH_UNDEFINED ? undefined : result;
	  }
	  return hasOwnProperty.call(data, key) ? data[key] : undefined;
	}

	/**
	 * Checks if a hash value for `key` exists.
	 *
	 * @private
	 * @name has
	 * @memberOf Hash
	 * @param {string} key The key of the entry to check.
	 * @returns {boolean} Returns `true` if an entry for `key` exists, else `false`.
	 */
	function hashHas(key) {
	  var data = this.__data__;
	  return nativeCreate ? data[key] !== undefined : hasOwnProperty.call(data, key);
	}

	/**
	 * Sets the hash `key` to `value`.
	 *
	 * @private
	 * @name set
	 * @memberOf Hash
	 * @param {string} key The key of the value to set.
	 * @param {*} value The value to set.
	 * @returns {Object} Returns the hash instance.
	 */
	function hashSet(key, value) {
	  var data = this.__data__;
	  data[key] = (nativeCreate && value === undefined) ? HASH_UNDEFINED : value;
	  return this;
	}

	// Add methods to `Hash`.
	Hash.prototype.clear = hashClear;
	Hash.prototype['delete'] = hashDelete;
	Hash.prototype.get = hashGet;
	Hash.prototype.has = hashHas;
	Hash.prototype.set = hashSet;

	/**
	 * Creates an list cache object.
	 *
	 * @private
	 * @constructor
	 * @param {Array} [entries] The key-value pairs to cache.
	 */
	function ListCache(entries) {
	  var index = -1,
	      length = entries ? entries.length : 0;

	  this.clear();
	  while (++index < length) {
	    var entry = entries[index];
	    this.set(entry[0], entry[1]);
	  }
	}

	/**
	 * Removes all key-value entries from the list cache.
	 *
	 * @private
	 * @name clear
	 * @memberOf ListCache
	 */
	function listCacheClear() {
	  this.__data__ = [];
	}

	/**
	 * Removes `key` and its value from the list cache.
	 *
	 * @private
	 * @name delete
	 * @memberOf ListCache
	 * @param {string} key The key of the value to remove.
	 * @returns {boolean} Returns `true` if the entry was removed, else `false`.
	 */
	function listCacheDelete(key) {
	  var data = this.__data__,
	      index = assocIndexOf(data, key);

	  if (index < 0) {
	    return false;
	  }
	  var lastIndex = data.length - 1;
	  if (index == lastIndex) {
	    data.pop();
	  } else {
	    splice.call(data, index, 1);
	  }
	  return true;
	}

	/**
	 * Gets the list cache value for `key`.
	 *
	 * @private
	 * @name get
	 * @memberOf ListCache
	 * @param {string} key The key of the value to get.
	 * @returns {*} Returns the entry value.
	 */
	function listCacheGet(key) {
	  var data = this.__data__,
	      index = assocIndexOf(data, key);

	  return index < 0 ? undefined : data[index][1];
	}

	/**
	 * Checks if a list cache value for `key` exists.
	 *
	 * @private
	 * @name has
	 * @memberOf ListCache
	 * @param {string} key The key of the entry to check.
	 * @returns {boolean} Returns `true` if an entry for `key` exists, else `false`.
	 */
	function listCacheHas(key) {
	  return assocIndexOf(this.__data__, key) > -1;
	}

	/**
	 * Sets the list cache `key` to `value`.
	 *
	 * @private
	 * @name set
	 * @memberOf ListCache
	 * @param {string} key The key of the value to set.
	 * @param {*} value The value to set.
	 * @returns {Object} Returns the list cache instance.
	 */
	function listCacheSet(key, value) {
	  var data = this.__data__,
	      index = assocIndexOf(data, key);

	  if (index < 0) {
	    data.push([key, value]);
	  } else {
	    data[index][1] = value;
	  }
	  return this;
	}

	// Add methods to `ListCache`.
	ListCache.prototype.clear = listCacheClear;
	ListCache.prototype['delete'] = listCacheDelete;
	ListCache.prototype.get = listCacheGet;
	ListCache.prototype.has = listCacheHas;
	ListCache.prototype.set = listCacheSet;

	/**
	 * Creates a map cache object to store key-value pairs.
	 *
	 * @private
	 * @constructor
	 * @param {Array} [entries] The key-value pairs to cache.
	 */
	function MapCache(entries) {
	  var index = -1,
	      length = entries ? entries.length : 0;

	  this.clear();
	  while (++index < length) {
	    var entry = entries[index];
	    this.set(entry[0], entry[1]);
	  }
	}

	/**
	 * Removes all key-value entries from the map.
	 *
	 * @private
	 * @name clear
	 * @memberOf MapCache
	 */
	function mapCacheClear() {
	  this.__data__ = {
	    'hash': new Hash,
	    'map': new (Map || ListCache),
	    'string': new Hash
	  };
	}

	/**
	 * Removes `key` and its value from the map.
	 *
	 * @private
	 * @name delete
	 * @memberOf MapCache
	 * @param {string} key The key of the value to remove.
	 * @returns {boolean} Returns `true` if the entry was removed, else `false`.
	 */
	function mapCacheDelete(key) {
	  return getMapData(this, key)['delete'](key);
	}

	/**
	 * Gets the map value for `key`.
	 *
	 * @private
	 * @name get
	 * @memberOf MapCache
	 * @param {string} key The key of the value to get.
	 * @returns {*} Returns the entry value.
	 */
	function mapCacheGet(key) {
	  return getMapData(this, key).get(key);
	}

	/**
	 * Checks if a map value for `key` exists.
	 *
	 * @private
	 * @name has
	 * @memberOf MapCache
	 * @param {string} key The key of the entry to check.
	 * @returns {boolean} Returns `true` if an entry for `key` exists, else `false`.
	 */
	function mapCacheHas(key) {
	  return getMapData(this, key).has(key);
	}

	/**
	 * Sets the map `key` to `value`.
	 *
	 * @private
	 * @name set
	 * @memberOf MapCache
	 * @param {string} key The key of the value to set.
	 * @param {*} value The value to set.
	 * @returns {Object} Returns the map cache instance.
	 */
	function mapCacheSet(key, value) {
	  getMapData(this, key).set(key, value);
	  return this;
	}

	// Add methods to `MapCache`.
	MapCache.prototype.clear = mapCacheClear;
	MapCache.prototype['delete'] = mapCacheDelete;
	MapCache.prototype.get = mapCacheGet;
	MapCache.prototype.has = mapCacheHas;
	MapCache.prototype.set = mapCacheSet;

	/**
	 * Creates a stack cache object to store key-value pairs.
	 *
	 * @private
	 * @constructor
	 * @param {Array} [entries] The key-value pairs to cache.
	 */
	function Stack(entries) {
	  this.__data__ = new ListCache(entries);
	}

	/**
	 * Removes all key-value entries from the stack.
	 *
	 * @private
	 * @name clear
	 * @memberOf Stack
	 */
	function stackClear() {
	  this.__data__ = new ListCache;
	}

	/**
	 * Removes `key` and its value from the stack.
	 *
	 * @private
	 * @name delete
	 * @memberOf Stack
	 * @param {string} key The key of the value to remove.
	 * @returns {boolean} Returns `true` if the entry was removed, else `false`.
	 */
	function stackDelete(key) {
	  return this.__data__['delete'](key);
	}

	/**
	 * Gets the stack value for `key`.
	 *
	 * @private
	 * @name get
	 * @memberOf Stack
	 * @param {string} key The key of the value to get.
	 * @returns {*} Returns the entry value.
	 */
	function stackGet(key) {
	  return this.__data__.get(key);
	}

	/**
	 * Checks if a stack value for `key` exists.
	 *
	 * @private
	 * @name has
	 * @memberOf Stack
	 * @param {string} key The key of the entry to check.
	 * @returns {boolean} Returns `true` if an entry for `key` exists, else `false`.
	 */
	function stackHas(key) {
	  return this.__data__.has(key);
	}

	/**
	 * Sets the stack `key` to `value`.
	 *
	 * @private
	 * @name set
	 * @memberOf Stack
	 * @param {string} key The key of the value to set.
	 * @param {*} value The value to set.
	 * @returns {Object} Returns the stack cache instance.
	 */
	function stackSet(key, value) {
	  var cache = this.__data__;
	  if (cache instanceof ListCache) {
	    var pairs = cache.__data__;
	    if (!Map || (pairs.length < LARGE_ARRAY_SIZE - 1)) {
	      pairs.push([key, value]);
	      return this;
	    }
	    cache = this.__data__ = new MapCache(pairs);
	  }
	  cache.set(key, value);
	  return this;
	}

	// Add methods to `Stack`.
	Stack.prototype.clear = stackClear;
	Stack.prototype['delete'] = stackDelete;
	Stack.prototype.get = stackGet;
	Stack.prototype.has = stackHas;
	Stack.prototype.set = stackSet;

	/**
	 * Creates an array of the enumerable property names of the array-like `value`.
	 *
	 * @private
	 * @param {*} value The value to query.
	 * @param {boolean} inherited Specify returning inherited property names.
	 * @returns {Array} Returns the array of property names.
	 */
	function arrayLikeKeys(value, inherited) {
	  // Safari 8.1 makes `arguments.callee` enumerable in strict mode.
	  // Safari 9 makes `arguments.length` enumerable in strict mode.
	  var result = (isArray(value) || isArguments(value))
	    ? baseTimes(value.length, String)
	    : [];

	  var length = result.length,
	      skipIndexes = !!length;

	  for (var key in value) {
	    if ((inherited || hasOwnProperty.call(value, key)) &&
	        !(skipIndexes && (key == 'length' || isIndex(key, length)))) {
	      result.push(key);
	    }
	  }
	  return result;
	}

	/**
	 * Assigns `value` to `key` of `object` if the existing value is not equivalent
	 * using [`SameValueZero`](http://ecma-international.org/ecma-262/7.0/#sec-samevaluezero)
	 * for equality comparisons.
	 *
	 * @private
	 * @param {Object} object The object to modify.
	 * @param {string} key The key of the property to assign.
	 * @param {*} value The value to assign.
	 */
	function assignValue(object, key, value) {
	  var objValue = object[key];
	  if (!(hasOwnProperty.call(object, key) && eq(objValue, value)) ||
	      (value === undefined && !(key in object))) {
	    object[key] = value;
	  }
	}

	/**
	 * Gets the index at which the `key` is found in `array` of key-value pairs.
	 *
	 * @private
	 * @param {Array} array The array to inspect.
	 * @param {*} key The key to search for.
	 * @returns {number} Returns the index of the matched value, else `-1`.
	 */
	function assocIndexOf(array, key) {
	  var length = array.length;
	  while (length--) {
	    if (eq(array[length][0], key)) {
	      return length;
	    }
	  }
	  return -1;
	}

	/**
	 * The base implementation of `_.assign` without support for multiple sources
	 * or `customizer` functions.
	 *
	 * @private
	 * @param {Object} object The destination object.
	 * @param {Object} source The source object.
	 * @returns {Object} Returns `object`.
	 */
	function baseAssign(object, source) {
	  return object && copyObject(source, keys(source), object);
	}

	/**
	 * The base implementation of `_.clone` and `_.cloneDeep` which tracks
	 * traversed objects.
	 *
	 * @private
	 * @param {*} value The value to clone.
	 * @param {boolean} [isDeep] Specify a deep clone.
	 * @param {boolean} [isFull] Specify a clone including symbols.
	 * @param {Function} [customizer] The function to customize cloning.
	 * @param {string} [key] The key of `value`.
	 * @param {Object} [object] The parent object of `value`.
	 * @param {Object} [stack] Tracks traversed objects and their clone counterparts.
	 * @returns {*} Returns the cloned value.
	 */
	function baseClone(value, isDeep, isFull, customizer, key, object, stack) {
	  var result;
	  if (customizer) {
	    result = object ? customizer(value, key, object, stack) : customizer(value);
	  }
	  if (result !== undefined) {
	    return result;
	  }
	  if (!isObject(value)) {
	    return value;
	  }
	  var isArr = isArray(value);
	  if (isArr) {
	    result = initCloneArray(value);
	    if (!isDeep) {
	      return copyArray(value, result);
	    }
	  } else {
	    var tag = getTag(value),
	        isFunc = tag == funcTag || tag == genTag;

	    if (isBuffer(value)) {
	      return cloneBuffer(value, isDeep);
	    }
	    if (tag == objectTag || tag == argsTag || (isFunc && !object)) {
	      if (isHostObject(value)) {
	        return object ? value : {};
	      }
	      result = initCloneObject(isFunc ? {} : value);
	      if (!isDeep) {
	        return copySymbols(value, baseAssign(result, value));
	      }
	    } else {
	      if (!cloneableTags[tag]) {
	        return object ? value : {};
	      }
	      result = initCloneByTag(value, tag, baseClone, isDeep);
	    }
	  }
	  // Check for circular references and return its corresponding clone.
	  stack || (stack = new Stack);
	  var stacked = stack.get(value);
	  if (stacked) {
	    return stacked;
	  }
	  stack.set(value, result);

	  if (!isArr) {
	    var props = isFull ? getAllKeys(value) : keys(value);
	  }
	  arrayEach(props || value, function(subValue, key) {
	    if (props) {
	      key = subValue;
	      subValue = value[key];
	    }
	    // Recursively populate clone (susceptible to call stack limits).
	    assignValue(result, key, baseClone(subValue, isDeep, isFull, customizer, key, value, stack));
	  });
	  return result;
	}

	/**
	 * The base implementation of `_.create` without support for assigning
	 * properties to the created object.
	 *
	 * @private
	 * @param {Object} prototype The object to inherit from.
	 * @returns {Object} Returns the new object.
	 */
	function baseCreate(proto) {
	  return isObject(proto) ? objectCreate(proto) : {};
	}

	/**
	 * The base implementation of `getAllKeys` and `getAllKeysIn` which uses
	 * `keysFunc` and `symbolsFunc` to get the enumerable property names and
	 * symbols of `object`.
	 *
	 * @private
	 * @param {Object} object The object to query.
	 * @param {Function} keysFunc The function to get the keys of `object`.
	 * @param {Function} symbolsFunc The function to get the symbols of `object`.
	 * @returns {Array} Returns the array of property names and symbols.
	 */
	function baseGetAllKeys(object, keysFunc, symbolsFunc) {
	  var result = keysFunc(object);
	  return isArray(object) ? result : arrayPush(result, symbolsFunc(object));
	}

	/**
	 * The base implementation of `getTag`.
	 *
	 * @private
	 * @param {*} value The value to query.
	 * @returns {string} Returns the `toStringTag`.
	 */
	function baseGetTag(value) {
	  return objectToString.call(value);
	}

	/**
	 * The base implementation of `_.isNative` without bad shim checks.
	 *
	 * @private
	 * @param {*} value The value to check.
	 * @returns {boolean} Returns `true` if `value` is a native function,
	 *  else `false`.
	 */
	function baseIsNative(value) {
	  if (!isObject(value) || isMasked(value)) {
	    return false;
	  }
	  var pattern = (isFunction(value) || isHostObject(value)) ? reIsNative : reIsHostCtor;
	  return pattern.test(toSource(value));
	}

	/**
	 * The base implementation of `_.keys` which doesn't treat sparse arrays as dense.
	 *
	 * @private
	 * @param {Object} object The object to query.
	 * @returns {Array} Returns the array of property names.
	 */
	function baseKeys(object) {
	  if (!isPrototype(object)) {
	    return nativeKeys(object);
	  }
	  var result = [];
	  for (var key in Object(object)) {
	    if (hasOwnProperty.call(object, key) && key != 'constructor') {
	      result.push(key);
	    }
	  }
	  return result;
	}

	/**
	 * Creates a clone of  `buffer`.
	 *
	 * @private
	 * @param {Buffer} buffer The buffer to clone.
	 * @param {boolean} [isDeep] Specify a deep clone.
	 * @returns {Buffer} Returns the cloned buffer.
	 */
	function cloneBuffer(buffer, isDeep) {
	  if (isDeep) {
	    return buffer.slice();
	  }
	  var result = new buffer.constructor(buffer.length);
	  buffer.copy(result);
	  return result;
	}

	/**
	 * Creates a clone of `arrayBuffer`.
	 *
	 * @private
	 * @param {ArrayBuffer} arrayBuffer The array buffer to clone.
	 * @returns {ArrayBuffer} Returns the cloned array buffer.
	 */
	function cloneArrayBuffer(arrayBuffer) {
	  var result = new arrayBuffer.constructor(arrayBuffer.byteLength);
	  new Uint8Array(result).set(new Uint8Array(arrayBuffer));
	  return result;
	}

	/**
	 * Creates a clone of `dataView`.
	 *
	 * @private
	 * @param {Object} dataView The data view to clone.
	 * @param {boolean} [isDeep] Specify a deep clone.
	 * @returns {Object} Returns the cloned data view.
	 */
	function cloneDataView(dataView, isDeep) {
	  var buffer = isDeep ? cloneArrayBuffer(dataView.buffer) : dataView.buffer;
	  return new dataView.constructor(buffer, dataView.byteOffset, dataView.byteLength);
	}

	/**
	 * Creates a clone of `map`.
	 *
	 * @private
	 * @param {Object} map The map to clone.
	 * @param {Function} cloneFunc The function to clone values.
	 * @param {boolean} [isDeep] Specify a deep clone.
	 * @returns {Object} Returns the cloned map.
	 */
	function cloneMap(map, isDeep, cloneFunc) {
	  var array = isDeep ? cloneFunc(mapToArray(map), true) : mapToArray(map);
	  return arrayReduce(array, addMapEntry, new map.constructor);
	}

	/**
	 * Creates a clone of `regexp`.
	 *
	 * @private
	 * @param {Object} regexp The regexp to clone.
	 * @returns {Object} Returns the cloned regexp.
	 */
	function cloneRegExp(regexp) {
	  var result = new regexp.constructor(regexp.source, reFlags.exec(regexp));
	  result.lastIndex = regexp.lastIndex;
	  return result;
	}

	/**
	 * Creates a clone of `set`.
	 *
	 * @private
	 * @param {Object} set The set to clone.
	 * @param {Function} cloneFunc The function to clone values.
	 * @param {boolean} [isDeep] Specify a deep clone.
	 * @returns {Object} Returns the cloned set.
	 */
	function cloneSet(set, isDeep, cloneFunc) {
	  var array = isDeep ? cloneFunc(setToArray(set), true) : setToArray(set);
	  return arrayReduce(array, addSetEntry, new set.constructor);
	}

	/**
	 * Creates a clone of the `symbol` object.
	 *
	 * @private
	 * @param {Object} symbol The symbol object to clone.
	 * @returns {Object} Returns the cloned symbol object.
	 */
	function cloneSymbol(symbol) {
	  return symbolValueOf ? Object(symbolValueOf.call(symbol)) : {};
	}

	/**
	 * Creates a clone of `typedArray`.
	 *
	 * @private
	 * @param {Object} typedArray The typed array to clone.
	 * @param {boolean} [isDeep] Specify a deep clone.
	 * @returns {Object} Returns the cloned typed array.
	 */
	function cloneTypedArray(typedArray, isDeep) {
	  var buffer = isDeep ? cloneArrayBuffer(typedArray.buffer) : typedArray.buffer;
	  return new typedArray.constructor(buffer, typedArray.byteOffset, typedArray.length);
	}

	/**
	 * Copies the values of `source` to `array`.
	 *
	 * @private
	 * @param {Array} source The array to copy values from.
	 * @param {Array} [array=[]] The array to copy values to.
	 * @returns {Array} Returns `array`.
	 */
	function copyArray(source, array) {
	  var index = -1,
	      length = source.length;

	  array || (array = Array(length));
	  while (++index < length) {
	    array[index] = source[index];
	  }
	  return array;
	}

	/**
	 * Copies properties of `source` to `object`.
	 *
	 * @private
	 * @param {Object} source The object to copy properties from.
	 * @param {Array} props The property identifiers to copy.
	 * @param {Object} [object={}] The object to copy properties to.
	 * @param {Function} [customizer] The function to customize copied values.
	 * @returns {Object} Returns `object`.
	 */
	function copyObject(source, props, object, customizer) {
	  object || (object = {});

	  var index = -1,
	      length = props.length;

	  while (++index < length) {
	    var key = props[index];

	    var newValue = customizer
	      ? customizer(object[key], source[key], key, object, source)
	      : undefined;

	    assignValue(object, key, newValue === undefined ? source[key] : newValue);
	  }
	  return object;
	}

	/**
	 * Copies own symbol properties of `source` to `object`.
	 *
	 * @private
	 * @param {Object} source The object to copy symbols from.
	 * @param {Object} [object={}] The object to copy symbols to.
	 * @returns {Object} Returns `object`.
	 */
	function copySymbols(source, object) {
	  return copyObject(source, getSymbols(source), object);
	}

	/**
	 * Creates an array of own enumerable property names and symbols of `object`.
	 *
	 * @private
	 * @param {Object} object The object to query.
	 * @returns {Array} Returns the array of property names and symbols.
	 */
	function getAllKeys(object) {
	  return baseGetAllKeys(object, keys, getSymbols);
	}

	/**
	 * Gets the data for `map`.
	 *
	 * @private
	 * @param {Object} map The map to query.
	 * @param {string} key The reference key.
	 * @returns {*} Returns the map data.
	 */
	function getMapData(map, key) {
	  var data = map.__data__;
	  return isKeyable(key)
	    ? data[typeof key == 'string' ? 'string' : 'hash']
	    : data.map;
	}

	/**
	 * Gets the native function at `key` of `object`.
	 *
	 * @private
	 * @param {Object} object The object to query.
	 * @param {string} key The key of the method to get.
	 * @returns {*} Returns the function if it's native, else `undefined`.
	 */
	function getNative(object, key) {
	  var value = getValue(object, key);
	  return baseIsNative(value) ? value : undefined;
	}

	/**
	 * Creates an array of the own enumerable symbol properties of `object`.
	 *
	 * @private
	 * @param {Object} object The object to query.
	 * @returns {Array} Returns the array of symbols.
	 */
	var getSymbols = nativeGetSymbols ? overArg(nativeGetSymbols, Object) : stubArray;

	/**
	 * Gets the `toStringTag` of `value`.
	 *
	 * @private
	 * @param {*} value The value to query.
	 * @returns {string} Returns the `toStringTag`.
	 */
	var getTag = baseGetTag;

	// Fallback for data views, maps, sets, and weak maps in IE 11,
	// for data views in Edge < 14, and promises in Node.js.
	if ((DataView && getTag(new DataView(new ArrayBuffer(1))) != dataViewTag) ||
	    (Map && getTag(new Map) != mapTag) ||
	    (Promise && getTag(Promise.resolve()) != promiseTag) ||
	    (Set && getTag(new Set) != setTag) ||
	    (WeakMap && getTag(new WeakMap) != weakMapTag)) {
	  getTag = function(value) {
	    var result = objectToString.call(value),
	        Ctor = result == objectTag ? value.constructor : undefined,
	        ctorString = Ctor ? toSource(Ctor) : undefined;

	    if (ctorString) {
	      switch (ctorString) {
	        case dataViewCtorString: return dataViewTag;
	        case mapCtorString: return mapTag;
	        case promiseCtorString: return promiseTag;
	        case setCtorString: return setTag;
	        case weakMapCtorString: return weakMapTag;
	      }
	    }
	    return result;
	  };
	}

	/**
	 * Initializes an array clone.
	 *
	 * @private
	 * @param {Array} array The array to clone.
	 * @returns {Array} Returns the initialized clone.
	 */
	function initCloneArray(array) {
	  var length = array.length,
	      result = array.constructor(length);

	  // Add properties assigned by `RegExp#exec`.
	  if (length && typeof array[0] == 'string' && hasOwnProperty.call(array, 'index')) {
	    result.index = array.index;
	    result.input = array.input;
	  }
	  return result;
	}

	/**
	 * Initializes an object clone.
	 *
	 * @private
	 * @param {Object} object The object to clone.
	 * @returns {Object} Returns the initialized clone.
	 */
	function initCloneObject(object) {
	  return (typeof object.constructor == 'function' && !isPrototype(object))
	    ? baseCreate(getPrototype(object))
	    : {};
	}

	/**
	 * Initializes an object clone based on its `toStringTag`.
	 *
	 * **Note:** This function only supports cloning values with tags of
	 * `Boolean`, `Date`, `Error`, `Number`, `RegExp`, or `String`.
	 *
	 * @private
	 * @param {Object} object The object to clone.
	 * @param {string} tag The `toStringTag` of the object to clone.
	 * @param {Function} cloneFunc The function to clone values.
	 * @param {boolean} [isDeep] Specify a deep clone.
	 * @returns {Object} Returns the initialized clone.
	 */
	function initCloneByTag(object, tag, cloneFunc, isDeep) {
	  var Ctor = object.constructor;
	  switch (tag) {
	    case arrayBufferTag:
	      return cloneArrayBuffer(object);

	    case boolTag:
	    case dateTag:
	      return new Ctor(+object);

	    case dataViewTag:
	      return cloneDataView(object, isDeep);

	    case float32Tag: case float64Tag:
	    case int8Tag: case int16Tag: case int32Tag:
	    case uint8Tag: case uint8ClampedTag: case uint16Tag: case uint32Tag:
	      return cloneTypedArray(object, isDeep);

	    case mapTag:
	      return cloneMap(object, isDeep, cloneFunc);

	    case numberTag:
	    case stringTag:
	      return new Ctor(object);

	    case regexpTag:
	      return cloneRegExp(object);

	    case setTag:
	      return cloneSet(object, isDeep, cloneFunc);

	    case symbolTag:
	      return cloneSymbol(object);
	  }
	}

	/**
	 * Checks if `value` is a valid array-like index.
	 *
	 * @private
	 * @param {*} value The value to check.
	 * @param {number} [length=MAX_SAFE_INTEGER] The upper bounds of a valid index.
	 * @returns {boolean} Returns `true` if `value` is a valid index, else `false`.
	 */
	function isIndex(value, length) {
	  length = length == null ? MAX_SAFE_INTEGER : length;
	  return !!length &&
	    (typeof value == 'number' || reIsUint.test(value)) &&
	    (value > -1 && value % 1 == 0 && value < length);
	}

	/**
	 * Checks if `value` is suitable for use as unique object key.
	 *
	 * @private
	 * @param {*} value The value to check.
	 * @returns {boolean} Returns `true` if `value` is suitable, else `false`.
	 */
	function isKeyable(value) {
	  var type = typeof value;
	  return (type == 'string' || type == 'number' || type == 'symbol' || type == 'boolean')
	    ? (value !== '__proto__')
	    : (value === null);
	}

	/**
	 * Checks if `func` has its source masked.
	 *
	 * @private
	 * @param {Function} func The function to check.
	 * @returns {boolean} Returns `true` if `func` is masked, else `false`.
	 */
	function isMasked(func) {
	  return !!maskSrcKey && (maskSrcKey in func);
	}

	/**
	 * Checks if `value` is likely a prototype object.
	 *
	 * @private
	 * @param {*} value The value to check.
	 * @returns {boolean} Returns `true` if `value` is a prototype, else `false`.
	 */
	function isPrototype(value) {
	  var Ctor = value && value.constructor,
	      proto = (typeof Ctor == 'function' && Ctor.prototype) || objectProto;

	  return value === proto;
	}

	/**
	 * Converts `func` to its source code.
	 *
	 * @private
	 * @param {Function} func The function to process.
	 * @returns {string} Returns the source code.
	 */
	function toSource(func) {
	  if (func != null) {
	    try {
	      return funcToString.call(func);
	    } catch (e) {}
	    try {
	      return (func + '');
	    } catch (e) {}
	  }
	  return '';
	}

	/**
	 * This method is like `_.clone` except that it recursively clones `value`.
	 *
	 * @static
	 * @memberOf _
	 * @since 1.0.0
	 * @category Lang
	 * @param {*} value The value to recursively clone.
	 * @returns {*} Returns the deep cloned value.
	 * @see _.clone
	 * @example
	 *
	 * var objects = [{ 'a': 1 }, { 'b': 2 }];
	 *
	 * var deep = _.cloneDeep(objects);
	 * console.log(deep[0] === objects[0]);
	 * // => false
	 */
	function cloneDeep(value) {
	  return baseClone(value, true, true);
	}

	/**
	 * Performs a
	 * [`SameValueZero`](http://ecma-international.org/ecma-262/7.0/#sec-samevaluezero)
	 * comparison between two values to determine if they are equivalent.
	 *
	 * @static
	 * @memberOf _
	 * @since 4.0.0
	 * @category Lang
	 * @param {*} value The value to compare.
	 * @param {*} other The other value to compare.
	 * @returns {boolean} Returns `true` if the values are equivalent, else `false`.
	 * @example
	 *
	 * var object = { 'a': 1 };
	 * var other = { 'a': 1 };
	 *
	 * _.eq(object, object);
	 * // => true
	 *
	 * _.eq(object, other);
	 * // => false
	 *
	 * _.eq('a', 'a');
	 * // => true
	 *
	 * _.eq('a', Object('a'));
	 * // => false
	 *
	 * _.eq(NaN, NaN);
	 * // => true
	 */
	function eq(value, other) {
	  return value === other || (value !== value && other !== other);
	}

	/**
	 * Checks if `value` is likely an `arguments` object.
	 *
	 * @static
	 * @memberOf _
	 * @since 0.1.0
	 * @category Lang
	 * @param {*} value The value to check.
	 * @returns {boolean} Returns `true` if `value` is an `arguments` object,
	 *  else `false`.
	 * @example
	 *
	 * _.isArguments(function() { return arguments; }());
	 * // => true
	 *
	 * _.isArguments([1, 2, 3]);
	 * // => false
	 */
	function isArguments(value) {
	  // Safari 8.1 makes `arguments.callee` enumerable in strict mode.
	  return isArrayLikeObject(value) && hasOwnProperty.call(value, 'callee') &&
	    (!propertyIsEnumerable.call(value, 'callee') || objectToString.call(value) == argsTag);
	}

	/**
	 * Checks if `value` is classified as an `Array` object.
	 *
	 * @static
	 * @memberOf _
	 * @since 0.1.0
	 * @category Lang
	 * @param {*} value The value to check.
	 * @returns {boolean} Returns `true` if `value` is an array, else `false`.
	 * @example
	 *
	 * _.isArray([1, 2, 3]);
	 * // => true
	 *
	 * _.isArray(document.body.children);
	 * // => false
	 *
	 * _.isArray('abc');
	 * // => false
	 *
	 * _.isArray(_.noop);
	 * // => false
	 */
	var isArray = Array.isArray;

	/**
	 * Checks if `value` is array-like. A value is considered array-like if it's
	 * not a function and has a `value.length` that's an integer greater than or
	 * equal to `0` and less than or equal to `Number.MAX_SAFE_INTEGER`.
	 *
	 * @static
	 * @memberOf _
	 * @since 4.0.0
	 * @category Lang
	 * @param {*} value The value to check.
	 * @returns {boolean} Returns `true` if `value` is array-like, else `false`.
	 * @example
	 *
	 * _.isArrayLike([1, 2, 3]);
	 * // => true
	 *
	 * _.isArrayLike(document.body.children);
	 * // => true
	 *
	 * _.isArrayLike('abc');
	 * // => true
	 *
	 * _.isArrayLike(_.noop);
	 * // => false
	 */
	function isArrayLike(value) {
	  return value != null && isLength(value.length) && !isFunction(value);
	}

	/**
	 * This method is like `_.isArrayLike` except that it also checks if `value`
	 * is an object.
	 *
	 * @static
	 * @memberOf _
	 * @since 4.0.0
	 * @category Lang
	 * @param {*} value The value to check.
	 * @returns {boolean} Returns `true` if `value` is an array-like object,
	 *  else `false`.
	 * @example
	 *
	 * _.isArrayLikeObject([1, 2, 3]);
	 * // => true
	 *
	 * _.isArrayLikeObject(document.body.children);
	 * // => true
	 *
	 * _.isArrayLikeObject('abc');
	 * // => false
	 *
	 * _.isArrayLikeObject(_.noop);
	 * // => false
	 */
	function isArrayLikeObject(value) {
	  return isObjectLike(value) && isArrayLike(value);
	}

	/**
	 * Checks if `value` is a buffer.
	 *
	 * @static
	 * @memberOf _
	 * @since 4.3.0
	 * @category Lang
	 * @param {*} value The value to check.
	 * @returns {boolean} Returns `true` if `value` is a buffer, else `false`.
	 * @example
	 *
	 * _.isBuffer(new Buffer(2));
	 * // => true
	 *
	 * _.isBuffer(new Uint8Array(2));
	 * // => false
	 */
	var isBuffer = nativeIsBuffer || stubFalse;

	/**
	 * Checks if `value` is classified as a `Function` object.
	 *
	 * @static
	 * @memberOf _
	 * @since 0.1.0
	 * @category Lang
	 * @param {*} value The value to check.
	 * @returns {boolean} Returns `true` if `value` is a function, else `false`.
	 * @example
	 *
	 * _.isFunction(_);
	 * // => true
	 *
	 * _.isFunction(/abc/);
	 * // => false
	 */
	function isFunction(value) {
	  // The use of `Object#toString` avoids issues with the `typeof` operator
	  // in Safari 8-9 which returns 'object' for typed array and other constructors.
	  var tag = isObject(value) ? objectToString.call(value) : '';
	  return tag == funcTag || tag == genTag;
	}

	/**
	 * Checks if `value` is a valid array-like length.
	 *
	 * **Note:** This method is loosely based on
	 * [`ToLength`](http://ecma-international.org/ecma-262/7.0/#sec-tolength).
	 *
	 * @static
	 * @memberOf _
	 * @since 4.0.0
	 * @category Lang
	 * @param {*} value The value to check.
	 * @returns {boolean} Returns `true` if `value` is a valid length, else `false`.
	 * @example
	 *
	 * _.isLength(3);
	 * // => true
	 *
	 * _.isLength(Number.MIN_VALUE);
	 * // => false
	 *
	 * _.isLength(Infinity);
	 * // => false
	 *
	 * _.isLength('3');
	 * // => false
	 */
	function isLength(value) {
	  return typeof value == 'number' &&
	    value > -1 && value % 1 == 0 && value <= MAX_SAFE_INTEGER;
	}

	/**
	 * Checks if `value` is the
	 * [language type](http://www.ecma-international.org/ecma-262/7.0/#sec-ecmascript-language-types)
	 * of `Object`. (e.g. arrays, functions, objects, regexes, `new Number(0)`, and `new String('')`)
	 *
	 * @static
	 * @memberOf _
	 * @since 0.1.0
	 * @category Lang
	 * @param {*} value The value to check.
	 * @returns {boolean} Returns `true` if `value` is an object, else `false`.
	 * @example
	 *
	 * _.isObject({});
	 * // => true
	 *
	 * _.isObject([1, 2, 3]);
	 * // => true
	 *
	 * _.isObject(_.noop);
	 * // => true
	 *
	 * _.isObject(null);
	 * // => false
	 */
	function isObject(value) {
	  var type = typeof value;
	  return !!value && (type == 'object' || type == 'function');
	}

	/**
	 * Checks if `value` is object-like. A value is object-like if it's not `null`
	 * and has a `typeof` result of "object".
	 *
	 * @static
	 * @memberOf _
	 * @since 4.0.0
	 * @category Lang
	 * @param {*} value The value to check.
	 * @returns {boolean} Returns `true` if `value` is object-like, else `false`.
	 * @example
	 *
	 * _.isObjectLike({});
	 * // => true
	 *
	 * _.isObjectLike([1, 2, 3]);
	 * // => true
	 *
	 * _.isObjectLike(_.noop);
	 * // => false
	 *
	 * _.isObjectLike(null);
	 * // => false
	 */
	function isObjectLike(value) {
	  return !!value && typeof value == 'object';
	}

	/**
	 * Creates an array of the own enumerable property names of `object`.
	 *
	 * **Note:** Non-object values are coerced to objects. See the
	 * [ES spec](http://ecma-international.org/ecma-262/7.0/#sec-object.keys)
	 * for more details.
	 *
	 * @static
	 * @since 0.1.0
	 * @memberOf _
	 * @category Object
	 * @param {Object} object The object to query.
	 * @returns {Array} Returns the array of property names.
	 * @example
	 *
	 * function Foo() {
	 *   this.a = 1;
	 *   this.b = 2;
	 * }
	 *
	 * Foo.prototype.c = 3;
	 *
	 * _.keys(new Foo);
	 * // => ['a', 'b'] (iteration order is not guaranteed)
	 *
	 * _.keys('hi');
	 * // => ['0', '1']
	 */
	function keys(object) {
	  return isArrayLike(object) ? arrayLikeKeys(object) : baseKeys(object);
	}

	/**
	 * This method returns a new empty array.
	 *
	 * @static
	 * @memberOf _
	 * @since 4.13.0
	 * @category Util
	 * @returns {Array} Returns the new empty array.
	 * @example
	 *
	 * var arrays = _.times(2, _.stubArray);
	 *
	 * console.log(arrays);
	 * // => [[], []]
	 *
	 * console.log(arrays[0] === arrays[1]);
	 * // => false
	 */
	function stubArray() {
	  return [];
	}

	/**
	 * This method returns `false`.
	 *
	 * @static
	 * @memberOf _
	 * @since 4.13.0
	 * @category Util
	 * @returns {boolean} Returns `false`.
	 * @example
	 *
	 * _.times(2, _.stubFalse);
	 * // => [false, false]
	 */
	function stubFalse() {
	  return false;
	}

	module.exports = cloneDeep;
	}(lodash_clonedeep, lodash_clonedeep.exports));

	var cloneDeep = lodash_clonedeep.exports;

	function dispatch(node, type, detail) {
	  detail = detail || {};
	  var document = node.ownerDocument, event = document.defaultView.CustomEvent;
	  if (typeof event === "function") {
	    event = new event(type, {detail: detail});
	  } else {
	    event = document.createEvent("Event");
	    event.initEvent(type, false, false);
	    event.detail = detail;
	  }
	  node.dispatchEvent(event);
	}

	// TODO https://twitter.com/mbostock/status/702737065121742848
	function isarray(value) {
	  return Array.isArray(value)
	      || value instanceof Int8Array
	      || value instanceof Int16Array
	      || value instanceof Int32Array
	      || value instanceof Uint8Array
	      || value instanceof Uint8ClampedArray
	      || value instanceof Uint16Array
	      || value instanceof Uint32Array
	      || value instanceof Float32Array
	      || value instanceof Float64Array;
	}

	// Non-integer keys in arrays, e.g. [1, 2, 0.5: "value"].
	function isindex(key) {
	  return key === (key | 0) + "";
	}

	function inspectName(name) {
	  const n = document.createElement("span");
	  n.className = "observablehq--cellname";
	  n.textContent = `${name} = `;
	  return n;
	}

	const symbolToString = Symbol.prototype.toString;

	// Symbols do not coerce to strings; they must be explicitly converted.
	function formatSymbol(symbol) {
	  return symbolToString.call(symbol);
	}

	const {getOwnPropertySymbols, prototype: {hasOwnProperty: hasOwnProperty$1}} = Object;
	const {toStringTag} = Symbol;

	const FORBIDDEN = {};

	const symbolsof = getOwnPropertySymbols;

	function isown(object, key) {
	  return hasOwnProperty$1.call(object, key);
	}

	function tagof(object) {
	  return object[toStringTag]
	      || (object.constructor && object.constructor.name)
	      || "Object";
	}

	function valueof$1(object, key) {
	  try {
	    const value = object[key];
	    if (value) value.constructor; // Test for SecurityError.
	    return value;
	  } catch (ignore) {
	    return FORBIDDEN;
	  }
	}

	const SYMBOLS = [
	  { symbol: "@@__IMMUTABLE_INDEXED__@@", name: "Indexed", modifier: true },
	  { symbol: "@@__IMMUTABLE_KEYED__@@", name: "Keyed", modifier: true },
	  { symbol: "@@__IMMUTABLE_LIST__@@", name: "List", arrayish: true },
	  { symbol: "@@__IMMUTABLE_MAP__@@", name: "Map" },
	  {
	    symbol: "@@__IMMUTABLE_ORDERED__@@",
	    name: "Ordered",
	    modifier: true,
	    prefix: true
	  },
	  { symbol: "@@__IMMUTABLE_RECORD__@@", name: "Record" },
	  {
	    symbol: "@@__IMMUTABLE_SET__@@",
	    name: "Set",
	    arrayish: true,
	    setish: true
	  },
	  { symbol: "@@__IMMUTABLE_STACK__@@", name: "Stack", arrayish: true }
	];

	function immutableName(obj) {
	  try {
	    let symbols = SYMBOLS.filter(({ symbol }) => obj[symbol] === true);
	    if (!symbols.length) return;

	    const name = symbols.find(s => !s.modifier);
	    const prefix =
	      name.name === "Map" && symbols.find(s => s.modifier && s.prefix);

	    const arrayish = symbols.some(s => s.arrayish);
	    const setish = symbols.some(s => s.setish);

	    return {
	      name: `${prefix ? prefix.name : ""}${name.name}`,
	      symbols,
	      arrayish: arrayish && !setish,
	      setish
	    };
	  } catch (e) {
	    return null;
	  }
	}

	const {getPrototypeOf, getOwnPropertyDescriptors} = Object;
	const objectPrototype = getPrototypeOf({});

	function inspectExpanded(object, _, name, proto) {
	  let arrayish = isarray(object);
	  let tag, fields, next, n;

	  if (object instanceof Map) {
	    if (object instanceof object.constructor) {
	      tag = `Map(${object.size})`;
	      fields = iterateMap$1;
	    } else { // avoid incompatible receiver error for prototype
	      tag = "Map()";
	      fields = iterateObject$1;
	    }
	  } else if (object instanceof Set) {
	    if (object instanceof object.constructor) {
	      tag = `Set(${object.size})`;
	      fields = iterateSet$1;
	    } else { // avoid incompatible receiver error for prototype
	      tag = "Set()";
	      fields = iterateObject$1;
	    }
	  } else if (arrayish) {
	    tag = `${object.constructor.name}(${object.length})`;
	    fields = iterateArray$1;
	  } else if ((n = immutableName(object))) {
	    tag = `Immutable.${n.name}${n.name === "Record" ? "" : `(${object.size})`}`;
	    arrayish = n.arrayish;
	    fields = n.arrayish
	      ? iterateImArray$1
	      : n.setish
	      ? iterateImSet$1
	      : iterateImObject$1;
	  } else if (proto) {
	    tag = tagof(object);
	    fields = iterateProto;
	  } else {
	    tag = tagof(object);
	    fields = iterateObject$1;
	  }

	  const span = document.createElement("span");
	  span.className = "observablehq--expanded";
	  if (name) {
	    span.appendChild(inspectName(name));
	  }
	  const a = span.appendChild(document.createElement("a"));
	  a.innerHTML = `<svg width=8 height=8 class='observablehq--caret'>
    <path d='M4 7L0 1h8z' fill='currentColor' />
  </svg>`;
	  a.appendChild(document.createTextNode(`${tag}${arrayish ? " [" : " {"}`));
	  a.addEventListener("mouseup", function(event) {
	    event.stopPropagation();
	    replace(span, inspectCollapsed(object, null, name, proto));
	  });

	  fields = fields(object);
	  for (let i = 0; !(next = fields.next()).done && i < 20; ++i) {
	    span.appendChild(next.value);
	  }

	  if (!next.done) {
	    const a = span.appendChild(document.createElement("a"));
	    a.className = "observablehq--field";
	    a.style.display = "block";
	    a.appendChild(document.createTextNode(`  … more`));
	    a.addEventListener("mouseup", function(event) {
	      event.stopPropagation();
	      span.insertBefore(next.value, span.lastChild.previousSibling);
	      for (let i = 0; !(next = fields.next()).done && i < 19; ++i) {
	        span.insertBefore(next.value, span.lastChild.previousSibling);
	      }
	      if (next.done) span.removeChild(span.lastChild.previousSibling);
	      dispatch(span, "load");
	    });
	  }

	  span.appendChild(document.createTextNode(arrayish ? "]" : "}"));

	  return span;
	}

	function* iterateMap$1(map) {
	  for (const [key, value] of map) {
	    yield formatMapField$1(key, value);
	  }
	  yield* iterateObject$1(map);
	}

	function* iterateSet$1(set) {
	  for (const value of set) {
	    yield formatSetField(value);
	  }
	  yield* iterateObject$1(set);
	}

	function* iterateImSet$1(set) {
	  for (const value of set) {
	    yield formatSetField(value);
	  }
	}

	function* iterateArray$1(array) {
	  for (let i = 0, n = array.length; i < n; ++i) {
	    if (i in array) {
	      yield formatField$1(i, valueof$1(array, i), "observablehq--index");
	    }
	  }
	  for (const key in array) {
	    if (!isindex(key) && isown(array, key)) {
	      yield formatField$1(key, valueof$1(array, key), "observablehq--key");
	    }
	  }
	  for (const symbol of symbolsof(array)) {
	    yield formatField$1(
	      formatSymbol(symbol),
	      valueof$1(array, symbol),
	      "observablehq--symbol"
	    );
	  }
	}

	function* iterateImArray$1(array) {
	  let i1 = 0;
	  for (const n = array.size; i1 < n; ++i1) {
	    yield formatField$1(i1, array.get(i1), true);
	  }
	}

	function* iterateProto(object) {
	  for (const key in getOwnPropertyDescriptors(object)) {
	    yield formatField$1(key, valueof$1(object, key), "observablehq--key");
	  }
	  for (const symbol of symbolsof(object)) {
	    yield formatField$1(
	      formatSymbol(symbol),
	      valueof$1(object, symbol),
	      "observablehq--symbol"
	    );
	  }

	  const proto = getPrototypeOf(object);
	  if (proto && proto !== objectPrototype) {
	    yield formatPrototype(proto);
	  }
	}

	function* iterateObject$1(object) {
	  for (const key in object) {
	    if (isown(object, key)) {
	      yield formatField$1(key, valueof$1(object, key), "observablehq--key");
	    }
	  }
	  for (const symbol of symbolsof(object)) {
	    yield formatField$1(
	      formatSymbol(symbol),
	      valueof$1(object, symbol),
	      "observablehq--symbol"
	    );
	  }

	  const proto = getPrototypeOf(object);
	  if (proto && proto !== objectPrototype) {
	    yield formatPrototype(proto);
	  }
	}

	function* iterateImObject$1(object) {
	  for (const [key, value] of object) {
	    yield formatField$1(key, value, "observablehq--key");
	  }
	}

	function formatPrototype(value) {
	  const item = document.createElement("div");
	  const span = item.appendChild(document.createElement("span"));
	  item.className = "observablehq--field";
	  span.className = "observablehq--prototype-key";
	  span.textContent = `  <prototype>`;
	  item.appendChild(document.createTextNode(": "));
	  item.appendChild(inspect(value, undefined, undefined, undefined, true));
	  return item;
	}

	function formatField$1(key, value, className) {
	  const item = document.createElement("div");
	  const span = item.appendChild(document.createElement("span"));
	  item.className = "observablehq--field";
	  span.className = className;
	  span.textContent = `  ${key}`;
	  item.appendChild(document.createTextNode(": "));
	  item.appendChild(inspect(value));
	  return item;
	}

	function formatMapField$1(key, value) {
	  const item = document.createElement("div");
	  item.className = "observablehq--field";
	  item.appendChild(document.createTextNode("  "));
	  item.appendChild(inspect(key));
	  item.appendChild(document.createTextNode(" => "));
	  item.appendChild(inspect(value));
	  return item;
	}

	function formatSetField(value) {
	  const item = document.createElement("div");
	  item.className = "observablehq--field";
	  item.appendChild(document.createTextNode("  "));
	  item.appendChild(inspect(value));
	  return item;
	}

	function hasSelection(elem) {
	  const sel = window.getSelection();
	  return (
	    sel.type === "Range" &&
	    (sel.containsNode(elem, true) ||
	      sel.anchorNode.isSelfOrDescendant(elem) ||
	      sel.focusNode.isSelfOrDescendant(elem))
	  );
	}

	function inspectCollapsed(object, shallow, name, proto) {
	  let arrayish = isarray(object);
	  let tag, fields, next, n;

	  if (object instanceof Map) {
	    if (object instanceof object.constructor) {
	      tag = `Map(${object.size})`;
	      fields = iterateMap;
	    } else { // avoid incompatible receiver error for prototype
	      tag = "Map()";
	      fields = iterateObject;
	    }
	  } else if (object instanceof Set) {
	    if (object instanceof object.constructor) {
	      tag = `Set(${object.size})`;
	      fields = iterateSet;
	    } else { // avoid incompatible receiver error for prototype
	      tag = "Set()";
	      fields = iterateObject;
	    }
	  } else if (arrayish) {
	    tag = `${object.constructor.name}(${object.length})`;
	    fields = iterateArray;
	  } else if ((n = immutableName(object))) {
	    tag = `Immutable.${n.name}${n.name === 'Record' ? '' : `(${object.size})`}`;
	    arrayish = n.arrayish;
	    fields = n.arrayish ? iterateImArray : n.setish ? iterateImSet : iterateImObject;
	  } else {
	    tag = tagof(object);
	    fields = iterateObject;
	  }

	  if (shallow) {
	    const span = document.createElement("span");
	    span.className = "observablehq--shallow";
	    if (name) {
	      span.appendChild(inspectName(name));
	    }
	    span.appendChild(document.createTextNode(tag));
	    span.addEventListener("mouseup", function(event) {
	      if (hasSelection(span)) return;
	      event.stopPropagation();
	      replace(span, inspectCollapsed(object));
	    });
	    return span;
	  }

	  const span = document.createElement("span");
	  span.className = "observablehq--collapsed";
	  if (name) {
	    span.appendChild(inspectName(name));
	  }
	  const a = span.appendChild(document.createElement("a"));
	  a.innerHTML = `<svg width=8 height=8 class='observablehq--caret'>
    <path d='M7 4L1 8V0z' fill='currentColor' />
  </svg>`;
	  a.appendChild(document.createTextNode(`${tag}${arrayish ? " [" : " {"}`));
	  span.addEventListener("mouseup", function(event) {
	    if (hasSelection(span)) return;
	    event.stopPropagation();
	    replace(span, inspectExpanded(object, null, name, proto));
	  }, true);

	  fields = fields(object);
	  for (let i = 0; !(next = fields.next()).done && i < 20; ++i) {
	    if (i > 0) span.appendChild(document.createTextNode(", "));
	    span.appendChild(next.value);
	  }

	  if (!next.done) span.appendChild(document.createTextNode(", …"));
	  span.appendChild(document.createTextNode(arrayish ? "]" : "}"));

	  return span;
	}

	function* iterateMap(map) {
	  for (const [key, value] of map) {
	    yield formatMapField(key, value);
	  }
	  yield* iterateObject(map);
	}

	function* iterateSet(set) {
	  for (const value of set) {
	    yield inspect(value, true);
	  }
	  yield* iterateObject(set);
	}

	function* iterateImSet(set) {
	  for (const value of set) {
	    yield inspect(value, true);
	  }
	}

	function* iterateImArray(array) {
	  let i0 = -1, i1 = 0;
	  for (const n = array.size; i1 < n; ++i1) {
	    if (i1 > i0 + 1) yield formatEmpty(i1 - i0 - 1);
	    yield inspect(array.get(i1), true);
	    i0 = i1;
	  }
	  if (i1 > i0 + 1) yield formatEmpty(i1 - i0 - 1);
	}

	function* iterateArray(array) {
	  let i0 = -1, i1 = 0;
	  for (const n = array.length; i1 < n; ++i1) {
	    if (i1 in array) {
	      if (i1 > i0 + 1) yield formatEmpty(i1 - i0 - 1);
	      yield inspect(valueof$1(array, i1), true);
	      i0 = i1;
	    }
	  }
	  if (i1 > i0 + 1) yield formatEmpty(i1 - i0 - 1);
	  for (const key in array) {
	    if (!isindex(key) && isown(array, key)) {
	      yield formatField(key, valueof$1(array, key), "observablehq--key");
	    }
	  }
	  for (const symbol of symbolsof(array)) {
	    yield formatField(formatSymbol(symbol), valueof$1(array, symbol), "observablehq--symbol");
	  }
	}

	function* iterateObject(object) {
	  for (const key in object) {
	    if (isown(object, key)) {
	      yield formatField(key, valueof$1(object, key), "observablehq--key");
	    }
	  }
	  for (const symbol of symbolsof(object)) {
	    yield formatField(formatSymbol(symbol), valueof$1(object, symbol), "observablehq--symbol");
	  }
	}

	function* iterateImObject(object) {
	  for (const [key, value] of object) {
	    yield formatField(key, value, "observablehq--key");
	  }
	}

	function formatEmpty(e) {
	  const span = document.createElement("span");
	  span.className = "observablehq--empty";
	  span.textContent = e === 1 ? "empty" : `empty × ${e}`;
	  return span;
	}

	function formatField(key, value, className) {
	  const fragment = document.createDocumentFragment();
	  const span = fragment.appendChild(document.createElement("span"));
	  span.className = className;
	  span.textContent = key;
	  fragment.appendChild(document.createTextNode(": "));
	  fragment.appendChild(inspect(value, true));
	  return fragment;
	}

	function formatMapField(key, value) {
	  const fragment = document.createDocumentFragment();
	  fragment.appendChild(inspect(key, true));
	  fragment.appendChild(document.createTextNode(" => "));
	  fragment.appendChild(inspect(value, true));
	  return fragment;
	}

	function format(date, fallback) {
	  if (!(date instanceof Date)) date = new Date(+date);
	  if (isNaN(date)) return typeof fallback === "function" ? fallback(date) : fallback;
	  const hours = date.getUTCHours();
	  const minutes = date.getUTCMinutes();
	  const seconds = date.getUTCSeconds();
	  const milliseconds = date.getUTCMilliseconds();
	  return `${formatYear$1(date.getUTCFullYear())}-${pad$1(date.getUTCMonth() + 1, 2)}-${pad$1(date.getUTCDate(), 2)}${
    hours || minutes || seconds || milliseconds ? `T${pad$1(hours, 2)}:${pad$1(minutes, 2)}${
      seconds || milliseconds ? `:${pad$1(seconds, 2)}${
        milliseconds ? `.${pad$1(milliseconds, 3)}` : ``
      }` : ``
    }Z` : ``
  }`;
	}

	function formatYear$1(year) {
	  return year < 0 ? `-${pad$1(-year, 6)}`
	    : year > 9999 ? `+${pad$1(year, 6)}`
	    : pad$1(year, 4);
	}

	function pad$1(value, width) {
	  return `${value}`.padStart(width, "0");
	}

	function formatDate$2(date) {
	  return format(date, "Invalid Date");
	}

	var errorToString = Error.prototype.toString;

	function formatError(value) {
	  return value.stack || errorToString.call(value);
	}

	var regExpToString = RegExp.prototype.toString;

	function formatRegExp(value) {
	  return regExpToString.call(value);
	}

	/* eslint-disable no-control-regex */
	const NEWLINE_LIMIT = 20;

	function formatString(string, shallow, expanded, name) {
	  if (shallow === false) {
	    // String has fewer escapes displayed with double quotes
	    if (count$1(string, /["\n]/g) <= count$1(string, /`|\${/g)) {
	      const span = document.createElement("span");
	      if (name) span.appendChild(inspectName(name));
	      const textValue = span.appendChild(document.createElement("span"));
	      textValue.className = "observablehq--string";
	      textValue.textContent = JSON.stringify(string);
	      return span;
	    }
	    const lines = string.split("\n");
	    if (lines.length > NEWLINE_LIMIT && !expanded) {
	      const div = document.createElement("div");
	      if (name) div.appendChild(inspectName(name));
	      const textValue = div.appendChild(document.createElement("span"));
	      textValue.className = "observablehq--string";
	      textValue.textContent = "`" + templatify(lines.slice(0, NEWLINE_LIMIT).join("\n"));
	      const splitter = div.appendChild(document.createElement("span"));
	      const truncatedCount = lines.length - NEWLINE_LIMIT;
	      splitter.textContent = `Show ${truncatedCount} truncated line${truncatedCount > 1 ? "s": ""}`; splitter.className = "observablehq--string-expand";
	      splitter.addEventListener("mouseup", function (event) {
	        event.stopPropagation();
	        replace(div, inspect(string, shallow, true, name));
	      });
	      return div;
	    }
	    const span = document.createElement("span");
	    if (name) span.appendChild(inspectName(name));
	    const textValue = span.appendChild(document.createElement("span"));
	    textValue.className = `observablehq--string${expanded ? " observablehq--expanded" : ""}`;
	    textValue.textContent = "`" + templatify(string) + "`";
	    return span;
	  }

	  const span = document.createElement("span");
	  if (name) span.appendChild(inspectName(name));
	  const textValue = span.appendChild(document.createElement("span"));
	  textValue.className = "observablehq--string";
	  textValue.textContent = JSON.stringify(string.length > 100 ?
	    `${string.slice(0, 50)}…${string.slice(-49)}` : string);
	  return span;
	}

	function templatify(string) {
	  return string.replace(/[\\`\x00-\x09\x0b-\x19]|\${/g, templatifyChar);
	}

	function templatifyChar(char) {
	  var code = char.charCodeAt(0);
	  switch (code) {
	    case 0x8: return "\\b";
	    case 0x9: return "\\t";
	    case 0xb: return "\\v";
	    case 0xc: return "\\f";
	    case 0xd: return "\\r";
	  }
	  return code < 0x10 ? "\\x0" + code.toString(16)
	      : code < 0x20 ? "\\x" + code.toString(16)
	      : "\\" + char;
	}

	function count$1(string, re) {
	  var n = 0;
	  while (re.exec(string)) ++n;
	  return n;
	}

	var toString$1 = Function.prototype.toString,
	    TYPE_ASYNC = {prefix: "async ƒ"},
	    TYPE_ASYNC_GENERATOR = {prefix: "async ƒ*"},
	    TYPE_CLASS = {prefix: "class"},
	    TYPE_FUNCTION = {prefix: "ƒ"},
	    TYPE_GENERATOR = {prefix: "ƒ*"};

	function inspectFunction(f, name) {
	  var type, m, t = toString$1.call(f);

	  switch (f.constructor && f.constructor.name) {
	    case "AsyncFunction": type = TYPE_ASYNC; break;
	    case "AsyncGeneratorFunction": type = TYPE_ASYNC_GENERATOR; break;
	    case "GeneratorFunction": type = TYPE_GENERATOR; break;
	    default: type = /^class\b/.test(t) ? TYPE_CLASS : TYPE_FUNCTION; break;
	  }

	  // A class, possibly named.
	  // class Name
	  if (type === TYPE_CLASS) {
	    return formatFunction(type, "", name);
	  }

	  // An arrow function with a single argument.
	  // foo =>
	  // async foo =>
	  if ((m = /^(?:async\s*)?(\w+)\s*=>/.exec(t))) {
	    return formatFunction(type, "(" + m[1] + ")", name);
	  }

	  // An arrow function with parenthesized arguments.
	  // (…)
	  // async (…)
	  if ((m = /^(?:async\s*)?\(\s*(\w+(?:\s*,\s*\w+)*)?\s*\)/.exec(t))) {
	    return formatFunction(type, m[1] ? "(" + m[1].replace(/\s*,\s*/g, ", ") + ")" : "()", name);
	  }

	  // A function, possibly: async, generator, anonymous, simply arguments.
	  // function name(…)
	  // function* name(…)
	  // async function name(…)
	  // async function* name(…)
	  if ((m = /^(?:async\s*)?function(?:\s*\*)?(?:\s*\w+)?\s*\(\s*(\w+(?:\s*,\s*\w+)*)?\s*\)/.exec(t))) {
	    return formatFunction(type, m[1] ? "(" + m[1].replace(/\s*,\s*/g, ", ") + ")" : "()", name);
	  }

	  // Something else, like destructuring, comments or default values.
	  return formatFunction(type, "(…)", name);
	}

	function formatFunction(type, args, cellname) {
	  var span = document.createElement("span");
	  span.className = "observablehq--function";
	  if (cellname) {
	    span.appendChild(inspectName(cellname));
	  }
	  var spanType = span.appendChild(document.createElement("span"));
	  spanType.className = "observablehq--keyword";
	  spanType.textContent = type.prefix;
	  span.appendChild(document.createTextNode(args));
	  return span;
	}

	const {prototype: {toString}} = Object;

	function inspect(value, shallow, expand, name, proto) {
	  let type = typeof value;
	  switch (type) {
	    case "boolean":
	    case "undefined": { value += ""; break; }
	    case "number": { value = value === 0 && 1 / value < 0 ? "-0" : value + ""; break; }
	    case "bigint": { value = value + "n"; break; }
	    case "symbol": { value = formatSymbol(value); break; }
	    case "function": { return inspectFunction(value, name); }
	    case "string": { return formatString(value, shallow, expand, name); }
	    default: {
	      if (value === null) { type = null, value = "null"; break; }
	      if (value instanceof Date) { type = "date", value = formatDate$2(value); break; }
	      if (value === FORBIDDEN) { type = "forbidden", value = "[forbidden]"; break; }
	      switch (toString.call(value)) {
	        case "[object RegExp]": { type = "regexp", value = formatRegExp(value); break; }
	        case "[object Error]": // https://github.com/lodash/lodash/blob/master/isError.js#L26
	        case "[object DOMException]": { type = "error", value = formatError(value); break; }
	        default: return (expand ? inspectExpanded : inspectCollapsed)(value, shallow, name, proto);
	      }
	      break;
	    }
	  }
	  const span = document.createElement("span");
	  if (name) span.appendChild(inspectName(name));
	  const n = span.appendChild(document.createElement("span"));
	  n.className = `observablehq--${type}`;
	  n.textContent = value;
	  return span;
	}

	function replace(spanOld, spanNew) {
	  if (spanOld.classList.contains("observablehq--inspect")) spanNew.classList.add("observablehq--inspect");
	  spanOld.parentNode.replaceChild(spanNew, spanOld);
	  dispatch(spanNew, "load");
	}

	const LOCATION_MATCH = /\s+\(\d+:\d+\)$/m;

	class Inspector {
	  constructor(node) {
	    if (!node) throw new Error("invalid node");
	    this._node = node;
	    node.classList.add("observablehq");
	  }
	  pending() {
	    const {_node} = this;
	    _node.classList.remove("observablehq--error");
	    _node.classList.add("observablehq--running");
	  }
	  fulfilled(value, name) {
	    const {_node} = this;
	    if (!isnode(value) || (value.parentNode && value.parentNode !== _node)) {
	      value = inspect(value, false, _node.firstChild // TODO Do this better.
	          && _node.firstChild.classList
	          && _node.firstChild.classList.contains("observablehq--expanded"), name);
	      value.classList.add("observablehq--inspect");
	    }
	    _node.classList.remove("observablehq--running", "observablehq--error");
	    if (_node.firstChild !== value) {
	      if (_node.firstChild) {
	        while (_node.lastChild !== _node.firstChild) _node.removeChild(_node.lastChild);
	        _node.replaceChild(value, _node.firstChild);
	      } else {
	        _node.appendChild(value);
	      }
	    }
	    dispatch(_node, "update");
	  }
	  rejected(error, name) {
	    const {_node} = this;
	    _node.classList.remove("observablehq--running");
	    _node.classList.add("observablehq--error");
	    while (_node.lastChild) _node.removeChild(_node.lastChild);
	    var div = document.createElement("div");
	    div.className = "observablehq--inspect";
	    if (name) div.appendChild(inspectName(name));
	    div.appendChild(document.createTextNode((error + "").replace(LOCATION_MATCH, "")));
	    _node.appendChild(div);
	    dispatch(_node, "error", {error: error});
	  }
	}

	Inspector.into = function(container) {
	  if (typeof container === "string") {
	    container = document.querySelector(container);
	    if (container == null) throw new Error("container not found");
	  }
	  return function() {
	    return new Inspector(container.appendChild(document.createElement("div")));
	  };
	};

	// Returns true if the given value is something that should be added to the DOM
	// by the inspector, rather than being inspected. This deliberately excludes
	// DocumentFragment since appending a fragment “dissolves” (mutates) the
	// fragment, and we wish for the inspector to not have side-effects. Also,
	// HTMLElement.prototype is an instanceof Element, but not an element!
	function isnode(value) {
	  return (value instanceof Element || value instanceof Text)
	      && (value instanceof value.constructor);
	}

	var EOL = {},
	    EOF = {},
	    QUOTE = 34,
	    NEWLINE = 10,
	    RETURN = 13;

	function objectConverter(columns) {
	  return new Function("d", "return {" + columns.map(function(name, i) {
	    return JSON.stringify(name) + ": d[" + i + "] || \"\"";
	  }).join(",") + "}");
	}

	function customConverter(columns, f) {
	  var object = objectConverter(columns);
	  return function(row, i) {
	    return f(object(row), i, columns);
	  };
	}

	// Compute unique columns in order of discovery.
	function inferColumns(rows) {
	  var columnSet = Object.create(null),
	      columns = [];

	  rows.forEach(function(row) {
	    for (var column in row) {
	      if (!(column in columnSet)) {
	        columns.push(columnSet[column] = column);
	      }
	    }
	  });

	  return columns;
	}

	function pad(value, width) {
	  var s = value + "", length = s.length;
	  return length < width ? new Array(width - length + 1).join(0) + s : s;
	}

	function formatYear(year) {
	  return year < 0 ? "-" + pad(-year, 6)
	    : year > 9999 ? "+" + pad(year, 6)
	    : pad(year, 4);
	}

	function formatDate$1(date) {
	  var hours = date.getUTCHours(),
	      minutes = date.getUTCMinutes(),
	      seconds = date.getUTCSeconds(),
	      milliseconds = date.getUTCMilliseconds();
	  return isNaN(date) ? "Invalid Date"
	      : formatYear(date.getUTCFullYear()) + "-" + pad(date.getUTCMonth() + 1, 2) + "-" + pad(date.getUTCDate(), 2)
	      + (milliseconds ? "T" + pad(hours, 2) + ":" + pad(minutes, 2) + ":" + pad(seconds, 2) + "." + pad(milliseconds, 3) + "Z"
	      : seconds ? "T" + pad(hours, 2) + ":" + pad(minutes, 2) + ":" + pad(seconds, 2) + "Z"
	      : minutes || hours ? "T" + pad(hours, 2) + ":" + pad(minutes, 2) + "Z"
	      : "");
	}

	function dsv$1(delimiter) {
	  var reFormat = new RegExp("[\"" + delimiter + "\n\r]"),
	      DELIMITER = delimiter.charCodeAt(0);

	  function parse(text, f) {
	    var convert, columns, rows = parseRows(text, function(row, i) {
	      if (convert) return convert(row, i - 1);
	      columns = row, convert = f ? customConverter(row, f) : objectConverter(row);
	    });
	    rows.columns = columns || [];
	    return rows;
	  }

	  function parseRows(text, f) {
	    var rows = [], // output rows
	        N = text.length,
	        I = 0, // current character index
	        n = 0, // current line number
	        t, // current token
	        eof = N <= 0, // current token followed by EOF?
	        eol = false; // current token followed by EOL?

	    // Strip the trailing newline.
	    if (text.charCodeAt(N - 1) === NEWLINE) --N;
	    if (text.charCodeAt(N - 1) === RETURN) --N;

	    function token() {
	      if (eof) return EOF;
	      if (eol) return eol = false, EOL;

	      // Unescape quotes.
	      var i, j = I, c;
	      if (text.charCodeAt(j) === QUOTE) {
	        while (I++ < N && text.charCodeAt(I) !== QUOTE || text.charCodeAt(++I) === QUOTE);
	        if ((i = I) >= N) eof = true;
	        else if ((c = text.charCodeAt(I++)) === NEWLINE) eol = true;
	        else if (c === RETURN) { eol = true; if (text.charCodeAt(I) === NEWLINE) ++I; }
	        return text.slice(j + 1, i - 1).replace(/""/g, "\"");
	      }

	      // Find next delimiter or newline.
	      while (I < N) {
	        if ((c = text.charCodeAt(i = I++)) === NEWLINE) eol = true;
	        else if (c === RETURN) { eol = true; if (text.charCodeAt(I) === NEWLINE) ++I; }
	        else if (c !== DELIMITER) continue;
	        return text.slice(j, i);
	      }

	      // Return last token before EOF.
	      return eof = true, text.slice(j, N);
	    }

	    while ((t = token()) !== EOF) {
	      var row = [];
	      while (t !== EOL && t !== EOF) row.push(t), t = token();
	      if (f && (row = f(row, n++)) == null) continue;
	      rows.push(row);
	    }

	    return rows;
	  }

	  function preformatBody(rows, columns) {
	    return rows.map(function(row) {
	      return columns.map(function(column) {
	        return formatValue(row[column]);
	      }).join(delimiter);
	    });
	  }

	  function format(rows, columns) {
	    if (columns == null) columns = inferColumns(rows);
	    return [columns.map(formatValue).join(delimiter)].concat(preformatBody(rows, columns)).join("\n");
	  }

	  function formatBody(rows, columns) {
	    if (columns == null) columns = inferColumns(rows);
	    return preformatBody(rows, columns).join("\n");
	  }

	  function formatRows(rows) {
	    return rows.map(formatRow).join("\n");
	  }

	  function formatRow(row) {
	    return row.map(formatValue).join(delimiter);
	  }

	  function formatValue(value) {
	    return value == null ? ""
	        : value instanceof Date ? formatDate$1(value)
	        : reFormat.test(value += "") ? "\"" + value.replace(/"/g, "\"\"") + "\""
	        : value;
	  }

	  return {
	    parse: parse,
	    parseRows: parseRows,
	    format: format,
	    formatBody: formatBody,
	    formatRows: formatRows,
	    formatRow: formatRow,
	    formatValue: formatValue
	  };
	}

	var csv = dsv$1(",");

	var csvParse = csv.parse;
	var csvParseRows = csv.parseRows;

	var tsv = dsv$1("\t");

	var tsvParse = tsv.parse;
	var tsvParseRows = tsv.parseRows;

	function autoType(object) {
	  for (var key in object) {
	    var value = object[key].trim(), number, m;
	    if (!value) value = null;
	    else if (value === "true") value = true;
	    else if (value === "false") value = false;
	    else if (value === "NaN") value = NaN;
	    else if (!isNaN(number = +value)) value = number;
	    else if (m = value.match(/^([-+]\d{2})?\d{4}(-\d{2}(-\d{2})?)?(T\d{2}:\d{2}(:\d{2}(\.\d{3})?)?(Z|[-+]\d{2}:\d{2})?)?$/)) {
	      if (fixtz && !!m[4] && !m[7]) value = value.replace(/-/g, "/").replace(/T/, " ");
	      value = new Date(value);
	    }
	    else continue;
	    object[key] = value;
	  }
	  return object;
	}

	// https://github.com/d3/d3-dsv/issues/45
	const fixtz = new Date("2019-01-01T00:00").getHours() || new Date("2019-07-01T00:00").getHours();

	const metas = new Map;
	const queue$1 = [];
	const map$2 = queue$1.map;
	const some = queue$1.some;
	const hasOwnProperty = queue$1.hasOwnProperty;
	const origin = "https://cdn.jsdelivr.net/npm/";
	const identifierRe = /^((?:@[^/@]+\/)?[^/@]+)(?:@([^/]+))?(?:\/(.*))?$/;
	const versionRe = /^\d+\.\d+\.\d+(-[\w-.+]+)?$/;
	const extensionRe = /\.[^/]*$/;
	const mains = ["unpkg", "jsdelivr", "browser", "main"];

	class RequireError extends Error {
	  constructor(message) {
	    super(message);
	  }
	}

	RequireError.prototype.name = RequireError.name;

	function main(meta) {
	  for (const key of mains) {
	    const value = meta[key];
	    if (typeof value === "string") {
	      return extensionRe.test(value) ? value : `${value}.js`;
	    }
	  }
	}

	function parseIdentifier(identifier) {
	  const match = identifierRe.exec(identifier);
	  return match && {
	    name: match[1],
	    version: match[2],
	    path: match[3]
	  };
	}

	function resolveMeta(target) {
	  const url = `${origin}${target.name}${target.version ? `@${target.version}` : ""}/package.json`;
	  let meta = metas.get(url);
	  if (!meta) metas.set(url, meta = fetch(url).then(response => {
	    if (!response.ok) throw new RequireError("unable to load package.json");
	    if (response.redirected && !metas.has(response.url)) metas.set(response.url, meta);
	    return response.json();
	  }));
	  return meta;
	}

	async function resolve$2(name, base) {
	  if (name.startsWith(origin)) name = name.substring(origin.length);
	  if (/^(\w+:)|\/\//i.test(name)) return name;
	  if (/^[.]{0,2}\//i.test(name)) return new URL(name, base == null ? location : base).href;
	  if (!name.length || /^[\s._]/.test(name) || /\s$/.test(name)) throw new RequireError("illegal name");
	  const target = parseIdentifier(name);
	  if (!target) return `${origin}${name}`;
	  if (!target.version && base != null && base.startsWith(origin)) {
	    const meta = await resolveMeta(parseIdentifier(base.substring(origin.length)));
	    target.version = meta.dependencies && meta.dependencies[target.name] || meta.peerDependencies && meta.peerDependencies[target.name];
	  }
	  if (target.path && !extensionRe.test(target.path)) target.path += ".js";
	  if (target.path && target.version && versionRe.test(target.version)) return `${origin}${target.name}@${target.version}/${target.path}`;
	  const meta = await resolveMeta(target);
	  return `${origin}${meta.name}@${meta.version}/${target.path || main(meta) || "index.js"}`;
	}

	var require = requireFrom(resolve$2);

	function requireFrom(resolver) {
	  const cache = new Map;
	  const requireBase = requireRelative(null);

	  function requireAbsolute(url) {
	    if (typeof url !== "string") return url;
	    let module = cache.get(url);
	    if (!module) cache.set(url, module = new Promise((resolve, reject) => {
	      const script = document.createElement("script");
	      script.onload = () => {
	        try { resolve(queue$1.pop()(requireRelative(url))); }
	        catch (error) { reject(new RequireError("invalid module")); }
	        script.remove();
	      };
	      script.onerror = () => {
	        reject(new RequireError("unable to load module"));
	        script.remove();
	      };
	      script.async = true;
	      script.src = url;
	      window.define = define$1;
	      document.head.appendChild(script);
	    }));
	    return module;
	  }

	  function requireRelative(base) {
	    return name => Promise.resolve(resolver(name, base)).then(requireAbsolute);
	  }

	  function requireAlias(aliases) {
	    return requireFrom((name, base) => {
	      if (name in aliases) {
	        name = aliases[name], base = null;
	        if (typeof name !== "string") return name;
	      }
	      return resolver(name, base);
	    });
	  }

	  function require(name) {
	    return arguments.length > 1
	        ? Promise.all(map$2.call(arguments, requireBase)).then(merge)
	        : requireBase(name);
	  }

	  require.alias = requireAlias;
	  require.resolve = resolver;

	  return require;
	}

	function merge(modules) {
	  const o = {};
	  for (const m of modules) {
	    for (const k in m) {
	      if (hasOwnProperty.call(m, k)) {
	        if (m[k] == null) Object.defineProperty(o, k, {get: getter(m, k)});
	        else o[k] = m[k];
	      }
	    }
	  }
	  return o;
	}

	function getter(object, name) {
	  return () => object[name];
	}

	function isbuiltin(name) {
	  name = name + "";
	  return name === "exports" || name === "module";
	}

	function define$1(name, dependencies, factory) {
	  const n = arguments.length;
	  if (n < 2) factory = name, dependencies = [];
	  else if (n < 3) factory = dependencies, dependencies = typeof name === "string" ? [] : name;
	  queue$1.push(some.call(dependencies, isbuiltin) ? require => {
	    const exports = {};
	    const module = {exports};
	    return Promise.all(map$2.call(dependencies, name => {
	      name = name + "";
	      return name === "exports" ? exports : name === "module" ? module : require(name);
	    })).then(dependencies => {
	      factory.apply(null, dependencies);
	      return module.exports;
	    });
	  } : require => {
	    return Promise.all(map$2.call(dependencies, require)).then(dependencies => {
	      return typeof factory === "function" ? factory.apply(null, dependencies) : factory;
	    });
	  });
	}

	define$1.amd = {};

	function dependency(name, version, main) {
	  return {
	    resolve(path = main) {
	      return `https://cdn.jsdelivr.net/npm/${name}@${version}/${path}`;
	    }
	  };
	}

	const d3 = dependency("d3", "7.1.0", "dist/d3.min.js");
	const inputs = dependency("@observablehq/inputs", "0.10.1", "dist/inputs.min.js");
	const plot = dependency("@observablehq/plot", "0.2.8", "dist/plot.umd.min.js");
	const graphviz = dependency("@observablehq/graphviz", "0.2.1", "dist/graphviz.min.js");
	const highlight = dependency("@observablehq/highlight.js", "2.0.0", "highlight.min.js");
	const katex = dependency("@observablehq/katex", "0.11.1", "dist/katex.min.js");
	const lodash = dependency("lodash", "4.17.21", "lodash.min.js");
	const htl = dependency("htl", "0.3.1", "dist/htl.min.js");
	const jszip = dependency("jszip", "3.7.1", "dist/jszip.min.js");
	const marked = dependency("marked", "0.3.12", "marked.min.js");
	const sql = dependency("sql.js", "1.6.1", "dist/sql-wasm.js");
	const vega = dependency("vega", "5.21.0", "build/vega.min.js");
	const vegalite = dependency("vega-lite", "5.1.1", "build/vega-lite.min.js");
	const vegaliteApi = dependency("vega-lite-api", "5.0.0", "build/vega-lite-api.min.js");
	const arrow = dependency("apache-arrow", "4.0.1", "Arrow.es2015.min.js");
	const arquero = dependency("arquero", "4.8.7", "dist/arquero.min.js");
	const topojson = dependency("topojson-client", "3.1.0", "dist/topojson-client.min.js");
	const exceljs = dependency("exceljs", "4.3.0", "dist/exceljs.min.js");

	async function sqlite(require) {
	  const init = await require(sql.resolve());
	  return init({locateFile: file => sql.resolve(`dist/${file}`)});
	}

	class SQLiteDatabaseClient {
	  constructor(db) {
	    Object.defineProperties(this, {
	      _db: {value: db}
	    });
	  }
	  static async open(source) {
	    const [SQL, buffer] = await Promise.all([sqlite(require), Promise.resolve(source).then(load$1)]);
	    return new SQLiteDatabaseClient(new SQL.Database(buffer));
	  }
	  async query(query, params) {
	    return await exec(this._db, query, params);
	  }
	  async queryRow(query, params) {
	    return (await this.query(query, params))[0] || null;
	  }
	  async explain(query, params) {
	    const rows = await this.query(`EXPLAIN QUERY PLAN ${query}`, params);
	    return element$1("pre", {className: "observablehq--inspect"}, [
	      text$2(rows.map(row => row.detail).join("\n"))
	    ]);
	  }
	  async describe(object) {
	    const rows = await (object === undefined
	      ? this.query(`SELECT name FROM sqlite_master WHERE type = 'table'`)
	      : this.query(`SELECT * FROM pragma_table_info(?)`, [object]));
	    if (!rows.length) throw new Error("Not found");
	    const {columns} = rows;
	    return element$1("table", {value: rows}, [
	      element$1("thead", [element$1("tr", columns.map(c => element$1("th", [text$2(c)])))]),
	      element$1("tbody", rows.map(r => element$1("tr", columns.map(c => element$1("td", [text$2(r[c])])))))
	    ]);
	  }
	}

	function load$1(source) {
	  return typeof source === "string" ? fetch(source).then(load$1)
	    : source instanceof Response || source instanceof Blob ? source.arrayBuffer().then(load$1)
	    : source instanceof ArrayBuffer ? new Uint8Array(source)
	    : source;
	}

	async function exec(db, query, params) {
	  const [result] = await db.exec(query, params);
	  if (!result) return [];
	  const {columns, values} = result;
	  const rows = values.map(row => Object.fromEntries(row.map((value, i) => [columns[i], value])));
	  rows.columns = columns;
	  return rows;
	}

	function element$1(name, props, children) {
	  if (arguments.length === 2) children = props, props = undefined;
	  const element = document.createElement(name);
	  if (props !== undefined) for (const p in props) element[p] = props[p];
	  if (children !== undefined) for (const c of children) element.appendChild(c);
	  return element;
	}

	function text$2(value) {
	  return document.createTextNode(value);
	}

	class Workbook {
	  constructor(workbook) {
	    Object.defineProperties(this, {
	      _: {value: workbook},
	      sheetNames: {
	        value: workbook.worksheets.map((s) => s.name),
	        enumerable: true,
	      },
	    });
	  }
	  sheet(name, options) {
	    const sname =
	      typeof name === "number"
	        ? this.sheetNames[name]
	        : this.sheetNames.includes((name += ""))
	        ? name
	        : null;
	    if (sname == null) throw new Error(`Sheet not found: ${name}`);
	    const sheet = this._.getWorksheet(sname);
	    return extract(sheet, options);
	  }
	}

	function extract(sheet, {range, headers} = {}) {
	  let [[c0, r0], [c1, r1]] = parseRange(range, sheet);
	  const headerRow = headers ? sheet._rows[r0++] : null;
	  let names = new Set(["#"]);
	  for (let n = c0; n <= c1; n++) {
	    const value = headerRow ? valueOf(headerRow.findCell(n + 1)) : null;
	    let name = (value && value + "") || toColumn(n);
	    while (names.has(name)) name += "_";
	    names.add(name);
	  }
	  names = new Array(c0).concat(Array.from(names));

	  const output = new Array(r1 - r0 + 1);
	  for (let r = r0; r <= r1; r++) {
	    const row = (output[r - r0] = Object.create(null, {"#": {value: r + 1}}));
	    const _row = sheet.getRow(r + 1);
	    if (_row.hasValues)
	      for (let c = c0; c <= c1; c++) {
	        const value = valueOf(_row.findCell(c + 1));
	        if (value != null) row[names[c + 1]] = value;
	      }
	  }

	  output.columns = names.filter(() => true); // Filter sparse columns
	  return output;
	}

	function valueOf(cell) {
	  if (!cell) return;
	  const {value} = cell;
	  if (value && typeof value === "object" && !(value instanceof Date)) {
	    if (value.formula || value.sharedFormula) {
	      return value.result && value.result.error ? NaN : value.result;
	    }
	    if (value.richText) {
	      return richText(value);
	    }
	    if (value.text) {
	      let {text} = value;
	      if (text.richText) text = richText(text);
	      return value.hyperlink && value.hyperlink !== text
	        ? `${value.hyperlink} ${text}`
	        : text;
	    }
	    return value;
	  }
	  return value;
	}

	function richText(value) {
	  return value.richText.map((d) => d.text).join("");
	}

	function parseRange(specifier = ":", {columnCount, rowCount}) {
	  specifier += "";
	  if (!specifier.match(/^[A-Z]*\d*:[A-Z]*\d*$/))
	    throw new Error("Malformed range specifier");
	  const [[c0 = 0, r0 = 0], [c1 = columnCount - 1, r1 = rowCount - 1]] =
	    specifier.split(":").map(fromCellReference);
	  return [
	    [c0, r0],
	    [c1, r1],
	  ];
	}

	// Returns the default column name for a zero-based column index.
	// For example: 0 -> "A", 1 -> "B", 25 -> "Z", 26 -> "AA", 27 -> "AB".
	function toColumn(c) {
	  let sc = "";
	  c++;
	  do {
	    sc = String.fromCharCode(64 + (c % 26 || 26)) + sc;
	  } while ((c = Math.floor((c - 1) / 26)));
	  return sc;
	}

	// Returns the zero-based indexes from a cell reference.
	// For example: "A1" -> [0, 0], "B2" -> [1, 1], "AA10" -> [26, 9].
	function fromCellReference(s) {
	  const [, sc, sr] = s.match(/^([A-Z]*)(\d*)$/);
	  let c = 0;
	  if (sc)
	    for (let i = 0; i < sc.length; i++)
	      c += Math.pow(26, sc.length - i - 1) * (sc.charCodeAt(i) - 64);
	  return [c ? c - 1 : undefined, sr ? +sr - 1 : undefined];
	}

	async function remote_fetch(file) {
	  const response = await fetch(await file.url());
	  if (!response.ok) throw new Error(`Unable to load file: ${file.name}`);
	  return response;
	}

	async function dsv(file, delimiter, {array = false, typed = false} = {}) {
	  const text = await file.text();
	  return (delimiter === "\t"
	      ? (array ? tsvParseRows : tsvParse)
	      : (array ? csvParseRows : csvParse))(text, typed && autoType);
	}

	class AbstractFile {
	  constructor(name) {
	    Object.defineProperty(this, "name", {value: name, enumerable: true});
	  }
	  async blob() {
	    return (await remote_fetch(this)).blob();
	  }
	  async arrayBuffer() {
	    return (await remote_fetch(this)).arrayBuffer();
	  }
	  async text() {
	    return (await remote_fetch(this)).text();
	  }
	  async json() {
	    return (await remote_fetch(this)).json();
	  }
	  async stream() {
	    return (await remote_fetch(this)).body;
	  }
	  async csv(options) {
	    return dsv(this, ",", options);
	  }
	  async tsv(options) {
	    return dsv(this, "\t", options);
	  }
	  async image() {
	    const url = await this.url();
	    return new Promise((resolve, reject) => {
	      const i = new Image;
	      if (new URL(url, document.baseURI).origin !== new URL(location).origin) {
	        i.crossOrigin = "anonymous";
	      }
	      i.onload = () => resolve(i);
	      i.onerror = () => reject(new Error(`Unable to load file: ${this.name}`));
	      i.src = url;
	    });
	  }
	  async arrow() {
	    const [Arrow, response] = await Promise.all([require(arrow.resolve()), remote_fetch(this)]);
	    return Arrow.Table.from(response);
	  }
	  async sqlite() {
	    return SQLiteDatabaseClient.open(remote_fetch(this));
	  }
	  async zip() {
	    const [JSZip, buffer] = await Promise.all([require(jszip.resolve()), this.arrayBuffer()]);
	    return new ZipArchive(await JSZip.loadAsync(buffer));
	  }
	  async xml(mimeType = "application/xml") {
	    return (new DOMParser).parseFromString(await this.text(), mimeType);
	  }
	  async html() {
	    return this.xml("text/html");
	  }
	  async xlsx() {
	    const [ExcelJS, buffer] = await Promise.all([require(exceljs.resolve()), this.arrayBuffer()]);
	    return new Workbook(await new ExcelJS.Workbook().xlsx.load(buffer));
	  }
	}

	class FileAttachment extends AbstractFile {
	  constructor(url, name) {
	    super(name);
	    Object.defineProperty(this, "_url", {value: url});
	  }
	  async url() {
	    return (await this._url) + "";
	  }
	}

	function NoFileAttachments(name) {
	  throw new Error(`File not found: ${name}`);
	}

	function FileAttachments(resolve) {
	  return Object.assign(
	    name => {
	      const url = resolve(name += ""); // Returns a Promise, string, or null.
	      if (url == null) throw new Error(`File not found: ${name}`);
	      return new FileAttachment(url, name);
	    },
	    {prototype: FileAttachment.prototype} // instanceof
	  );
	}

	class ZipArchive {
	  constructor(archive) {
	    Object.defineProperty(this, "_", {value: archive});
	    this.filenames = Object.keys(archive.files).filter(name => !archive.files[name].dir);
	  }
	  file(path) {
	    const object = this._.file(path += "");
	    if (!object || object.dir) throw new Error(`file not found: ${path}`);
	    return new ZipArchiveEntry(object);
	  }
	}

	class ZipArchiveEntry extends AbstractFile {
	  constructor(object) {
	    super(object.name);
	    Object.defineProperty(this, "_", {value: object});
	    Object.defineProperty(this, "_url", {writable: true});
	  }
	  async url() {
	    return this._url || (this._url = this.blob().then(URL.createObjectURL));
	  }
	  async blob() {
	    return this._.async("blob");
	  }
	  async arrayBuffer() {
	    return this._.async("arraybuffer");
	  }
	  async text() {
	    return this._.async("text");
	  }
	  async json() {
	    return JSON.parse(await this.text());
	  }
	}

	function canvas(width, height) {
	  var canvas = document.createElement("canvas");
	  canvas.width = width;
	  canvas.height = height;
	  return canvas;
	}

	function context2d(width, height, dpi) {
	  if (dpi == null) dpi = devicePixelRatio;
	  var canvas = document.createElement("canvas");
	  canvas.width = width * dpi;
	  canvas.height = height * dpi;
	  canvas.style.width = width + "px";
	  var context = canvas.getContext("2d");
	  context.scale(dpi, dpi);
	  return context;
	}

	function download(value, name = "untitled", label = "Save") {
	  const a = document.createElement("a");
	  const b = a.appendChild(document.createElement("button"));
	  b.textContent = label;
	  a.download = name;

	  async function reset() {
	    await new Promise(requestAnimationFrame);
	    URL.revokeObjectURL(a.href);
	    a.removeAttribute("href");
	    b.textContent = label;
	    b.disabled = false;
	  }

	  a.onclick = async event => {
	    b.disabled = true;
	    if (a.href) return reset(); // Already saved.
	    b.textContent = "Saving…";
	    try {
	      const object = await (typeof value === "function" ? value() : value);
	      b.textContent = "Download";
	      a.href = URL.createObjectURL(object); // eslint-disable-line require-atomic-updates
	    } catch (ignore) {
	      b.textContent = label;
	    }
	    if (event.eventPhase) return reset(); // Already downloaded.
	    b.disabled = false;
	  };

	  return a;
	}

	var namespaces = {
	  math: "http://www.w3.org/1998/Math/MathML",
	  svg: "http://www.w3.org/2000/svg",
	  xhtml: "http://www.w3.org/1999/xhtml",
	  xlink: "http://www.w3.org/1999/xlink",
	  xml: "http://www.w3.org/XML/1998/namespace",
	  xmlns: "http://www.w3.org/2000/xmlns/"
	};

	function element(name, attributes) {
	  var prefix = name += "", i = prefix.indexOf(":"), value;
	  if (i >= 0 && (prefix = name.slice(0, i)) !== "xmlns") name = name.slice(i + 1);
	  var element = namespaces.hasOwnProperty(prefix) // eslint-disable-line no-prototype-builtins
	      ? document.createElementNS(namespaces[prefix], name)
	      : document.createElement(name);
	  if (attributes) for (var key in attributes) {
	    prefix = key, i = prefix.indexOf(":"), value = attributes[key];
	    if (i >= 0 && (prefix = key.slice(0, i)) !== "xmlns") key = key.slice(i + 1);
	    if (namespaces.hasOwnProperty(prefix)) element.setAttributeNS(namespaces[prefix], key, value); // eslint-disable-line no-prototype-builtins
	    else element.setAttribute(key, value);
	  }
	  return element;
	}

	function input$1(type) {
	  var input = document.createElement("input");
	  if (type != null) input.type = type;
	  return input;
	}

	function range$2(min, max, step) {
	  if (arguments.length === 1) max = min, min = null;
	  var input = document.createElement("input");
	  input.min = min = min == null ? 0 : +min;
	  input.max = max = max == null ? 1 : +max;
	  input.step = step == null ? "any" : step = +step;
	  input.type = "range";
	  return input;
	}

	function select$1(values) {
	  var select = document.createElement("select");
	  Array.prototype.forEach.call(values, function(value) {
	    var option = document.createElement("option");
	    option.value = option.textContent = value;
	    select.appendChild(option);
	  });
	  return select;
	}

	function svg$1(width, height) {
	  var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
	  svg.setAttribute("viewBox", [0, 0, width, height]);
	  svg.setAttribute("width", width);
	  svg.setAttribute("height", height);
	  return svg;
	}

	function text$1(value) {
	  return document.createTextNode(value);
	}

	var count = 0;

	function uid(name) {
	  return new Id("O-" + (name == null ? "" : name + "-") + ++count);
	}

	function Id(id) {
	  this.id = id;
	  this.href = new URL(`#${id}`, location) + "";
	}

	Id.prototype.toString = function() {
	  return "url(" + this.href + ")";
	};

	var DOM = {
	  canvas: canvas,
	  context2d: context2d,
	  download: download,
	  element: element,
	  input: input$1,
	  range: range$2,
	  select: select$1,
	  svg: svg$1,
	  text: text$1,
	  uid: uid
	};

	function buffer(file) {
	  return new Promise(function(resolve, reject) {
	    var reader = new FileReader;
	    reader.onload = function() { resolve(reader.result); };
	    reader.onerror = reject;
	    reader.readAsArrayBuffer(file);
	  });
	}

	function text(file) {
	  return new Promise(function(resolve, reject) {
	    var reader = new FileReader;
	    reader.onload = function() { resolve(reader.result); };
	    reader.onerror = reject;
	    reader.readAsText(file);
	  });
	}

	function url(file) {
	  return new Promise(function(resolve, reject) {
	    var reader = new FileReader;
	    reader.onload = function() { resolve(reader.result); };
	    reader.onerror = reject;
	    reader.readAsDataURL(file);
	  });
	}

	var Files = {
	  buffer: buffer,
	  text: text,
	  url: url
	};

	function that() {
	  return this;
	}

	function disposable(value, dispose) {
	  let done = false;
	  if (typeof dispose !== "function") {
	    throw new Error("dispose is not a function");
	  }
	  return {
	    [Symbol.iterator]: that,
	    next: () => done ? {done: true} : (done = true, {done: false, value}),
	    return: () => (done = true, dispose(value), {done: true}),
	    throw: () => ({done: done = true})
	  };
	}

	function* filter(iterator, test) {
	  var result, index = -1;
	  while (!(result = iterator.next()).done) {
	    if (test(result.value, ++index)) {
	      yield result.value;
	    }
	  }
	}

	function observe(initialize) {
	  let stale = false;
	  let value;
	  let resolve;
	  const dispose = initialize(change);

	  if (dispose != null && typeof dispose !== "function") {
	    throw new Error(typeof dispose.then === "function"
	        ? "async initializers are not supported"
	        : "initializer returned something, but not a dispose function");
	  }

	  function change(x) {
	    if (resolve) resolve(x), resolve = null;
	    else stale = true;
	    return value = x;
	  }

	  function next() {
	    return {done: false, value: stale
	        ? (stale = false, Promise.resolve(value))
	        : new Promise(_ => (resolve = _))};
	  }

	  return {
	    [Symbol.iterator]: that,
	    throw: () => ({done: true}),
	    return: () => (dispose != null && dispose(), {done: true}),
	    next
	  };
	}

	function input(input) {
	  return observe(function(change) {
	    var event = eventof(input), value = valueof(input);
	    function inputted() { change(valueof(input)); }
	    input.addEventListener(event, inputted);
	    if (value !== undefined) change(value);
	    return function() { input.removeEventListener(event, inputted); };
	  });
	}

	function valueof(input) {
	  switch (input.type) {
	    case "range":
	    case "number": return input.valueAsNumber;
	    case "date": return input.valueAsDate;
	    case "checkbox": return input.checked;
	    case "file": return input.multiple ? input.files : input.files[0];
	    case "select-multiple": return Array.from(input.selectedOptions, o => o.value);
	    default: return input.value;
	  }
	}

	function eventof(input) {
	  switch (input.type) {
	    case "button":
	    case "submit":
	    case "checkbox": return "click";
	    case "file": return "change";
	    default: return "input";
	  }
	}

	function* map$1(iterator, transform) {
	  var result, index = -1;
	  while (!(result = iterator.next()).done) {
	    yield transform(result.value, ++index);
	  }
	}

	function queue(initialize) {
	  let resolve;
	  const queue = [];
	  const dispose = initialize(push);

	  if (dispose != null && typeof dispose !== "function") {
	    throw new Error(typeof dispose.then === "function"
	        ? "async initializers are not supported"
	        : "initializer returned something, but not a dispose function");
	  }

	  function push(x) {
	    queue.push(x);
	    if (resolve) resolve(queue.shift()), resolve = null;
	    return x;
	  }

	  function next() {
	    return {done: false, value: queue.length
	        ? Promise.resolve(queue.shift())
	        : new Promise(_ => (resolve = _))};
	  }

	  return {
	    [Symbol.iterator]: that,
	    throw: () => ({done: true}),
	    return: () => (dispose != null && dispose(), {done: true}),
	    next
	  };
	}

	function* range$1(start, stop, step) {
	  start = +start;
	  stop = +stop;
	  step = (n = arguments.length) < 2 ? (stop = start, start = 0, 1) : n < 3 ? 1 : +step;
	  var i = -1, n = Math.max(0, Math.ceil((stop - start) / step)) | 0;
	  while (++i < n) {
	    yield start + i * step;
	  }
	}

	function valueAt(iterator, i) {
	  if (!isFinite(i = +i) || i < 0 || i !== i | 0) return;
	  var result, index = -1;
	  while (!(result = iterator.next()).done) {
	    if (++index === i) {
	      return result.value;
	    }
	  }
	}

	function worker(source) {
	  const url = URL.createObjectURL(new Blob([source], {type: "text/javascript"}));
	  const worker = new Worker(url);
	  return disposable(worker, () => {
	    worker.terminate();
	    URL.revokeObjectURL(url);
	  });
	}

	var Generators = {
	  disposable: disposable,
	  filter: filter,
	  input: input,
	  map: map$1,
	  observe: observe,
	  queue: queue,
	  range: range$1,
	  valueAt: valueAt,
	  worker: worker
	};

	function template(render, wrapper) {
	  return function(strings) {
	    var string = strings[0],
	        parts = [], part,
	        root = null,
	        node, nodes,
	        walker,
	        i, n, j, m, k = -1;

	    // Concatenate the text using comments as placeholders.
	    for (i = 1, n = arguments.length; i < n; ++i) {
	      part = arguments[i];
	      if (part instanceof Node) {
	        parts[++k] = part;
	        string += "<!--o:" + k + "-->";
	      } else if (Array.isArray(part)) {
	        for (j = 0, m = part.length; j < m; ++j) {
	          node = part[j];
	          if (node instanceof Node) {
	            if (root === null) {
	              parts[++k] = root = document.createDocumentFragment();
	              string += "<!--o:" + k + "-->";
	            }
	            root.appendChild(node);
	          } else {
	            root = null;
	            string += node;
	          }
	        }
	        root = null;
	      } else {
	        string += part;
	      }
	      string += strings[i];
	    }

	    // Render the text.
	    root = render(string);

	    // Walk the rendered content to replace comment placeholders.
	    if (++k > 0) {
	      nodes = new Array(k);
	      walker = document.createTreeWalker(root, NodeFilter.SHOW_COMMENT, null, false);
	      while (walker.nextNode()) {
	        node = walker.currentNode;
	        if (/^o:/.test(node.nodeValue)) {
	          nodes[+node.nodeValue.slice(2)] = node;
	        }
	      }
	      for (i = 0; i < k; ++i) {
	        if (node = nodes[i]) {
	          node.parentNode.replaceChild(parts[i], node);
	        }
	      }
	    }

	    // Is the rendered content
	    // … a parent of a single child? Detach and return the child.
	    // … a document fragment? Replace the fragment with an element.
	    // … some other node? Return it.
	    return root.childNodes.length === 1 ? root.removeChild(root.firstChild)
	        : root.nodeType === 11 ? ((node = wrapper()).appendChild(root), node)
	        : root;
	  };
	}

	var html$1 = template(function(string) {
	  var template = document.createElement("template");
	  template.innerHTML = string.trim();
	  return document.importNode(template.content, true);
	}, function() {
	  return document.createElement("span");
	});

	function md(require) {
	  return require(marked.resolve()).then(function(marked) {
	    return template(
	      function(string) {
	        var root = document.createElement("div");
	        root.innerHTML = marked(string, {langPrefix: ""}).trim();
	        var code = root.querySelectorAll("pre code[class]");
	        if (code.length > 0) {
	          require(highlight.resolve()).then(function(hl) {
	            code.forEach(function(block) {
	              function done() {
	                hl.highlightBlock(block);
	                block.parentNode.classList.add("observablehq--md-pre");
	              }
	              if (hl.getLanguage(block.className)) {
	                done();
	              } else {
	                require(highlight.resolve("async-languages/index.js"))
	                  .then(index => {
	                    if (index.has(block.className)) {
	                      return require(highlight.resolve("async-languages/" + index.get(block.className))).then(language => {
	                        hl.registerLanguage(block.className, language);
	                      });
	                    }
	                  })
	                  .then(done, done);
	              }
	            });
	          });
	        }
	        return root;
	      },
	      function() {
	        return document.createElement("div");
	      }
	    );
	  });
	}

	function Mutable(value) {
	  let change;
	  Object.defineProperties(this, {
	    generator: {value: observe(_ => void (change = _))},
	    value: {get: () => value, set: x => change(value = x)} // eslint-disable-line no-setter-return
	  });
	  if (value !== undefined) change(value);
	}

	function* now() {
	  while (true) {
	    yield Date.now();
	  }
	}

	function delay(duration, value) {
	  return new Promise(function(resolve) {
	    setTimeout(function() {
	      resolve(value);
	    }, duration);
	  });
	}

	var timeouts = new Map;

	function timeout(now, time) {
	  var t = new Promise(function(resolve) {
	    timeouts.delete(time);
	    var delay = time - now;
	    if (!(delay > 0)) throw new Error("invalid time");
	    if (delay > 0x7fffffff) throw new Error("too long to wait");
	    setTimeout(resolve, delay);
	  });
	  timeouts.set(time, t);
	  return t;
	}

	function when(time, value) {
	  var now;
	  return (now = timeouts.get(time = +time)) ? now.then(() => value)
	      : (now = Date.now()) >= time ? Promise.resolve(value)
	      : timeout(now, time).then(() => value);
	}

	function tick(duration, value) {
	  return when(Math.ceil((Date.now() + 1) / duration) * duration, value);
	}

	var Promises = {
	  delay: delay,
	  tick: tick,
	  when: when
	};

	function resolve$1(name, base) {
	  if (/^(\w+:)|\/\//i.test(name)) return name;
	  if (/^[.]{0,2}\//i.test(name)) return new URL(name, base == null ? location : base).href;
	  if (!name.length || /^[\s._]/.test(name) || /\s$/.test(name)) throw new Error("illegal name");
	  return "https://unpkg.com/" + name;
	}

	function requirer(resolve) {
	  return resolve == null ? require : requireFrom(resolve);
	}

	var svg = template(function(string) {
	  var root = document.createElementNS("http://www.w3.org/2000/svg", "g");
	  root.innerHTML = string.trim();
	  return root;
	}, function() {
	  return document.createElementNS("http://www.w3.org/2000/svg", "g");
	});

	var raw = String.raw;

	function style(href) {
	  return new Promise(function(resolve, reject) {
	    var link = document.createElement("link");
	    link.rel = "stylesheet";
	    link.href = href;
	    link.onerror = reject;
	    link.onload = resolve;
	    document.head.appendChild(link);
	  });
	}

	function tex(require) {
	  return Promise.all([
	    require(katex.resolve()),
	    style(katex.resolve("dist/katex.min.css"))
	  ]).then(function(values) {
	    var katex = values[0], tex = renderer();

	    function renderer(options) {
	      return function() {
	        var root = document.createElement("div");
	        katex.render(raw.apply(String, arguments), root, options);
	        return root.removeChild(root.firstChild);
	      };
	    }

	    tex.options = renderer;
	    tex.block = renderer({displayMode: true});
	    return tex;
	  });
	}

	async function vl(require) {
	  const [v, vl, api] = await Promise.all([vega, vegalite, vegaliteApi].map(d => require(d.resolve())));
	  return api.register(v, vl);
	}

	function width$2() {
	  return observe(function(change) {
	    var width = change(document.body.clientWidth);
	    function resized() {
	      var w = document.body.clientWidth;
	      if (w !== width) change(width = w);
	    }
	    window.addEventListener("resize", resized);
	    return function() {
	      window.removeEventListener("resize", resized);
	    };
	  });
	}

	var Library = Object.assign(function Library(resolver) {
	  const require = requirer(resolver);
	  Object.defineProperties(this, properties({
	    FileAttachment: () => NoFileAttachments,
	    Arrow: () => require(arrow.resolve()),
	    Inputs: () => require(inputs.resolve()).then(Inputs => ({...Inputs, file: Inputs.fileOf(AbstractFile)})),
	    Mutable: () => Mutable,
	    Plot: () => require(plot.resolve()),
	    SQLite: () => sqlite(require),
	    SQLiteDatabaseClient: () => SQLiteDatabaseClient,
	    _: () => require(lodash.resolve()),
	    aq: () => require.alias({"apache-arrow": arrow.resolve()})(arquero.resolve()),
	    d3: () => require(d3.resolve()),
	    dot: () => require(graphviz.resolve()),
	    htl: () => require(htl.resolve()),
	    html: () => html$1,
	    md: () => md(require),
	    now,
	    require: () => require,
	    resolve: () => resolve$1,
	    svg: () => svg,
	    tex: () => tex(require),
	    topojson: () => require(topojson.resolve()),
	    vl: () => vl(require),
	    width: width$2,

	    // Note: these are namespace objects, and thus exposed directly rather than
	    // being wrapped in a function. This allows library.Generators to resolve,
	    // rather than needing module.value.
	    DOM,
	    Files,
	    Generators,
	    Promises
	  }));
	}, {resolve: require.resolve});

	function properties(values) {
	  return Object.fromEntries(Object.entries(values).map(property));
	}

	function property([key, value]) {
	  return [key, ({value, writable: true, enumerable: true})];
	}

	function RuntimeError(message, input) {
	  this.message = message + "";
	  this.input = input;
	}

	RuntimeError.prototype = Object.create(Error.prototype);
	RuntimeError.prototype.name = "RuntimeError";
	RuntimeError.prototype.constructor = RuntimeError;

	function generatorish(value) {
	  return value
	      && typeof value.next === "function"
	      && typeof value.return === "function";
	}

	function load(notebook, library, observer) {
	  if (typeof library == "function") observer = library, library = null;
	  if (typeof observer !== "function") throw new Error("invalid observer");
	  if (library == null) library = new Library();

	  const {modules, id} = notebook;
	  const map = new Map;
	  const runtime = new Runtime(library);
	  const main = runtime_module(id);

	  function runtime_module(id) {
	    let module = map.get(id);
	    if (!module) map.set(id, module = runtime.module());
	    return module;
	  }

	  for (const m of modules) {
	    const module = runtime_module(m.id);
	    let i = 0;
	    for (const v of m.variables) {
	      if (v.from) module.import(v.remote, v.name, runtime_module(v.from));
	      else if (module === main) module.variable(observer(v, i, m.variables)).define(v.name, v.inputs, v.value);
	      else module.define(v.name, v.inputs, v.value);
	      ++i;
	    }
	  }

	  return runtime;
	}

	var prototype = Array.prototype;
	var map = prototype.map;
	var forEach = prototype.forEach;

	function constant(x) {
	  return function() {
	    return x;
	  };
	}

	function identity$1(x) {
	  return x;
	}

	function rethrow(e) {
	  return function() {
	    throw e;
	  };
	}

	function noop() {}

	var TYPE_NORMAL = 1; // a normal variable
	var TYPE_IMPLICIT = 2; // created on reference
	var TYPE_DUPLICATE = 3; // created on duplicate definition

	var no_observer = {};

	function Variable(type, module, observer) {
	  if (!observer) observer = no_observer;
	  Object.defineProperties(this, {
	    _observer: {value: observer, writable: true},
	    _definition: {value: variable_undefined, writable: true},
	    _duplicate: {value: undefined, writable: true},
	    _duplicates: {value: undefined, writable: true},
	    _indegree: {value: NaN, writable: true}, // The number of computing inputs.
	    _inputs: {value: [], writable: true},
	    _invalidate: {value: noop, writable: true},
	    _module: {value: module},
	    _name: {value: null, writable: true},
	    _outputs: {value: new Set, writable: true},
	    _promise: {value: Promise.resolve(undefined), writable: true},
	    _reachable: {value: observer !== no_observer, writable: true}, // Is this variable transitively visible?
	    _rejector: {value: variable_rejector(this)},
	    _type: {value: type},
	    _value: {value: undefined, writable: true},
	    _version: {value: 0, writable: true}
	  });
	}

	Object.defineProperties(Variable.prototype, {
	  _pending: {value: variable_pending, writable: true, configurable: true},
	  _fulfilled: {value: variable_fulfilled, writable: true, configurable: true},
	  _rejected: {value: variable_rejected, writable: true, configurable: true},
	  define: {value: variable_define, writable: true, configurable: true},
	  delete: {value: variable_delete, writable: true, configurable: true},
	  import: {value: variable_import, writable: true, configurable: true}
	});

	function variable_attach(variable) {
	  variable._module._runtime._dirty.add(variable);
	  variable._outputs.add(this);
	}

	function variable_detach(variable) {
	  variable._module._runtime._dirty.add(variable);
	  variable._outputs.delete(this);
	}

	function variable_undefined() {
	  throw variable_undefined;
	}

	function variable_rejector(variable) {
	  return function(error) {
	    if (error === variable_undefined) throw new RuntimeError(variable._name + " is not defined", variable._name);
	    if (error instanceof Error && error.message) throw new RuntimeError(error.message, variable._name);
	    throw new RuntimeError(variable._name + " could not be resolved", variable._name);
	  };
	}

	function variable_duplicate(name) {
	  return function() {
	    throw new RuntimeError(name + " is defined more than once");
	  };
	}

	function variable_define(name, inputs, definition) {
	  switch (arguments.length) {
	    case 1: {
	      definition = name, name = inputs = null;
	      break;
	    }
	    case 2: {
	      definition = inputs;
	      if (typeof name === "string") inputs = null;
	      else inputs = name, name = null;
	      break;
	    }
	  }
	  return variable_defineImpl.call(this,
	    name == null ? null : name + "",
	    inputs == null ? [] : map.call(inputs, this._module._resolve, this._module),
	    typeof definition === "function" ? definition : constant(definition)
	  );
	}

	function variable_defineImpl(name, inputs, definition) {
	  var scope = this._module._scope, runtime = this._module._runtime;

	  this._inputs.forEach(variable_detach, this);
	  inputs.forEach(variable_attach, this);
	  this._inputs = inputs;
	  this._definition = definition;
	  this._value = undefined;

	  // Is this an active variable (that may require disposal)?
	  if (definition === noop) runtime._variables.delete(this);
	  else runtime._variables.add(this);

	  // Did the variable’s name change? Time to patch references!
	  if (name !== this._name || scope.get(name) !== this) {
	    var error, found;

	    if (this._name) { // Did this variable previously have a name?
	      if (this._outputs.size) { // And did other variables reference this variable?
	        scope.delete(this._name);
	        found = this._module._resolve(this._name);
	        found._outputs = this._outputs, this._outputs = new Set;
	        found._outputs.forEach(function(output) { output._inputs[output._inputs.indexOf(this)] = found; }, this);
	        found._outputs.forEach(runtime._updates.add, runtime._updates);
	        runtime._dirty.add(found).add(this);
	        scope.set(this._name, found);
	      } else if ((found = scope.get(this._name)) === this) { // Do no other variables reference this variable?
	        scope.delete(this._name); // It’s safe to delete!
	      } else if (found._type === TYPE_DUPLICATE) { // Do other variables assign this name?
	        found._duplicates.delete(this); // This variable no longer assigns this name.
	        this._duplicate = undefined;
	        if (found._duplicates.size === 1) { // Is there now only one variable assigning this name?
	          found = found._duplicates.keys().next().value; // Any references are now fixed!
	          error = scope.get(this._name);
	          found._outputs = error._outputs, error._outputs = new Set;
	          found._outputs.forEach(function(output) { output._inputs[output._inputs.indexOf(error)] = found; });
	          found._definition = found._duplicate, found._duplicate = undefined;
	          runtime._dirty.add(error).add(found);
	          runtime._updates.add(found);
	          scope.set(this._name, found);
	        }
	      } else {
	        throw new Error;
	      }
	    }

	    if (this._outputs.size) throw new Error;

	    if (name) { // Does this variable have a new name?
	      if (found = scope.get(name)) { // Do other variables reference or assign this name?
	        if (found._type === TYPE_DUPLICATE) { // Do multiple other variables already define this name?
	          this._definition = variable_duplicate(name), this._duplicate = definition;
	          found._duplicates.add(this);
	        } else if (found._type === TYPE_IMPLICIT) { // Are the variable references broken?
	          this._outputs = found._outputs, found._outputs = new Set; // Now they’re fixed!
	          this._outputs.forEach(function(output) { output._inputs[output._inputs.indexOf(found)] = this; }, this);
	          runtime._dirty.add(found).add(this);
	          scope.set(name, this);
	        } else { // Does another variable define this name?
	          found._duplicate = found._definition, this._duplicate = definition; // Now they’re duplicates.
	          error = new Variable(TYPE_DUPLICATE, this._module);
	          error._name = name;
	          error._definition = this._definition = found._definition = variable_duplicate(name);
	          error._outputs = found._outputs, found._outputs = new Set;
	          error._outputs.forEach(function(output) { output._inputs[output._inputs.indexOf(found)] = error; });
	          error._duplicates = new Set([this, found]);
	          runtime._dirty.add(found).add(error);
	          runtime._updates.add(found).add(error);
	          scope.set(name, error);
	        }
	      } else {
	        scope.set(name, this);
	      }
	    }

	    this._name = name;
	  }

	  runtime._updates.add(this);
	  runtime._compute();
	  return this;
	}

	function variable_import(remote, name, module) {
	  if (arguments.length < 3) module = name, name = remote;
	  return variable_defineImpl.call(this, name + "", [module._resolve(remote + "")], identity$1);
	}

	function variable_delete() {
	  return variable_defineImpl.call(this, null, [], noop);
	}

	function variable_pending() {
	  if (this._observer.pending) this._observer.pending();
	}

	function variable_fulfilled(value) {
	  if (this._observer.fulfilled) this._observer.fulfilled(value, this._name);
	}

	function variable_rejected(error) {
	  if (this._observer.rejected) this._observer.rejected(error, this._name);
	}

	function Module(runtime, builtins = []) {
	  Object.defineProperties(this, {
	    _runtime: {value: runtime},
	    _scope: {value: new Map},
	    _builtins: {value: new Map([
	      ["invalidation", variable_invalidation],
	      ["visibility", variable_visibility],
	      ...builtins
	    ])},
	    _source: {value: null, writable: true}
	  });
	}

	Object.defineProperties(Module.prototype, {
	  _copy: {value: module_copy, writable: true, configurable: true},
	  _resolve: {value: module_resolve, writable: true, configurable: true},
	  redefine: {value: module_redefine, writable: true, configurable: true},
	  define: {value: module_define, writable: true, configurable: true},
	  derive: {value: module_derive, writable: true, configurable: true},
	  import: {value: module_import, writable: true, configurable: true},
	  value: {value: module_value, writable: true, configurable: true},
	  variable: {value: module_variable, writable: true, configurable: true},
	  builtin: {value: module_builtin, writable: true, configurable: true}
	});

	function module_redefine(name) {
	  var v = this._scope.get(name);
	  if (!v) throw new RuntimeError(name + " is not defined");
	  if (v._type === TYPE_DUPLICATE) throw new RuntimeError(name + " is defined more than once");
	  return v.define.apply(v, arguments);
	}

	function module_define() {
	  var v = new Variable(TYPE_NORMAL, this);
	  return v.define.apply(v, arguments);
	}

	function module_import() {
	  var v = new Variable(TYPE_NORMAL, this);
	  return v.import.apply(v, arguments);
	}

	function module_variable(observer) {
	  return new Variable(TYPE_NORMAL, this, observer);
	}

	async function module_value(name) {
	  var v = this._scope.get(name);
	  if (!v) throw new RuntimeError(name + " is not defined");
	  if (v._observer === no_observer) {
	    v._observer = true;
	    this._runtime._dirty.add(v);
	  }
	  await this._runtime._compute();
	  return v._promise;
	}

	function module_derive(injects, injectModule) {
	  var copy = new Module(this._runtime, this._builtins);
	  copy._source = this;
	  forEach.call(injects, function(inject) {
	    if (typeof inject !== "object") inject = {name: inject + ""};
	    if (inject.alias == null) inject.alias = inject.name;
	    copy.import(inject.name, inject.alias, injectModule);
	  });
	  Promise.resolve().then(() => {
	    const modules = new Set([this]);
	    for (const module of modules) {
	      for (const variable of module._scope.values()) {
	        if (variable._definition === identity$1) { // import
	          const module = variable._inputs[0]._module;
	          const source = module._source || module;
	          if (source === this) { // circular import-with!
	            console.warn("circular module definition; ignoring"); // eslint-disable-line no-console
	            return;
	          }
	          modules.add(source);
	        }
	      }
	    }
	    this._copy(copy, new Map);
	  });
	  return copy;
	}

	function module_copy(copy, map) {
	  copy._source = this;
	  map.set(this, copy);
	  for (const [name, source] of this._scope) {
	    var target = copy._scope.get(name);
	    if (target && target._type === TYPE_NORMAL) continue; // injection
	    if (source._definition === identity$1) { // import
	      var sourceInput = source._inputs[0],
	          sourceModule = sourceInput._module;
	      copy.import(sourceInput._name, name, map.get(sourceModule)
	        || (sourceModule._source
	           ? sourceModule._copy(new Module(copy._runtime, copy._builtins), map) // import-with
	           : sourceModule));
	    } else {
	      copy.define(name, source._inputs.map(variable_name), source._definition);
	    }
	  }
	  return copy;
	}

	function module_resolve(name) {
	  var variable = this._scope.get(name), value;
	  if (!variable) {
	    variable = new Variable(TYPE_IMPLICIT, this);
	    if (this._builtins.has(name)) {
	      variable.define(name, constant(this._builtins.get(name)));
	    } else if (this._runtime._builtin._scope.has(name)) {
	      variable.import(name, this._runtime._builtin);
	    } else {
	      try {
	        value = this._runtime._global(name);
	      } catch (error) {
	        return variable.define(name, rethrow(error));
	      }
	      if (value === undefined) {
	        this._scope.set(variable._name = name, variable);
	      } else {
	        variable.define(name, constant(value));
	      }
	    }
	  }
	  return variable;
	}

	function module_builtin(name, value) {
	  this._builtins.set(name, value);
	}

	function variable_name(variable) {
	  return variable._name;
	}

	const frame = typeof requestAnimationFrame === "function" ? requestAnimationFrame : setImmediate;

	var variable_invalidation = {};
	var variable_visibility = {};

	function Runtime(builtins = new Library, global = window_global) {
	  var builtin = this.module();
	  Object.defineProperties(this, {
	    _dirty: {value: new Set},
	    _updates: {value: new Set},
	    _computing: {value: null, writable: true},
	    _init: {value: null, writable: true},
	    _modules: {value: new Map},
	    _variables: {value: new Set},
	    _disposed: {value: false, writable: true},
	    _builtin: {value: builtin},
	    _global: {value: global}
	  });
	  if (builtins) for (var name in builtins) {
	    (new Variable(TYPE_IMPLICIT, builtin)).define(name, [], builtins[name]);
	  }
	}

	Object.defineProperties(Runtime, {
	  load: {value: load, writable: true, configurable: true}
	});

	Object.defineProperties(Runtime.prototype, {
	  _compute: {value: runtime_compute, writable: true, configurable: true},
	  _computeSoon: {value: runtime_computeSoon, writable: true, configurable: true},
	  _computeNow: {value: runtime_computeNow, writable: true, configurable: true},
	  dispose: {value: runtime_dispose, writable: true, configurable: true},
	  module: {value: runtime_module, writable: true, configurable: true},
	  fileAttachments: {value: FileAttachments, writable: true, configurable: true}
	});

	function runtime_dispose() {
	  this._computing = Promise.resolve();
	  this._disposed = true;
	  this._variables.forEach(v => {
	    v._invalidate();
	    v._version = NaN;
	  });
	}

	function runtime_module(define, observer = noop) {
	  let module;
	  if (define === undefined) {
	    if (module = this._init) {
	      this._init = null;
	      return module;
	    }
	    return new Module(this);
	  }
	  module = this._modules.get(define);
	  if (module) return module;
	  this._init = module = new Module(this);
	  this._modules.set(define, module);
	  try {
	    define(this, observer);
	  } finally {
	    this._init = null;
	  }
	  return module;
	}

	function runtime_compute() {
	  return this._computing || (this._computing = this._computeSoon());
	}

	function runtime_computeSoon() {
	  var runtime = this;
	  return new Promise(function(resolve) {
	    frame(function() {
	      resolve();
	      runtime._disposed || runtime._computeNow();
	    });
	  });
	}

	function runtime_computeNow() {
	  var queue = [],
	      variables,
	      variable;

	  // Compute the reachability of the transitive closure of dirty variables.
	  // Any newly-reachable variable must also be recomputed.
	  // Any no-longer-reachable variable must be terminated.
	  variables = new Set(this._dirty);
	  variables.forEach(function(variable) {
	    variable._inputs.forEach(variables.add, variables);
	    const reachable = variable_reachable(variable);
	    if (reachable > variable._reachable) {
	      this._updates.add(variable);
	    } else if (reachable < variable._reachable) {
	      variable._invalidate();
	    }
	    variable._reachable = reachable;
	  }, this);

	  // Compute the transitive closure of updating, reachable variables.
	  variables = new Set(this._updates);
	  variables.forEach(function(variable) {
	    if (variable._reachable) {
	      variable._indegree = 0;
	      variable._outputs.forEach(variables.add, variables);
	    } else {
	      variable._indegree = NaN;
	      variables.delete(variable);
	    }
	  });

	  this._computing = null;
	  this._updates.clear();
	  this._dirty.clear();

	  // Compute the indegree of updating variables.
	  variables.forEach(function(variable) {
	    variable._outputs.forEach(variable_increment);
	  });

	  do {
	    // Identify the root variables (those with no updating inputs).
	    variables.forEach(function(variable) {
	      if (variable._indegree === 0) {
	        queue.push(variable);
	      }
	    });

	    // Compute the variables in topological order.
	    while (variable = queue.pop()) {
	      variable_compute(variable);
	      variable._outputs.forEach(postqueue);
	      variables.delete(variable);
	    }

	    // Any remaining variables are circular, or depend on them.
	    variables.forEach(function(variable) {
	      if (variable_circular(variable)) {
	        variable_error(variable, new RuntimeError("circular definition"));
	        variable._outputs.forEach(variable_decrement);
	        variables.delete(variable);
	      }
	    });
	  } while (variables.size);

	  function postqueue(variable) {
	    if (--variable._indegree === 0) {
	      queue.push(variable);
	    }
	  }
	}

	function variable_circular(variable) {
	  const inputs = new Set(variable._inputs);
	  for (const i of inputs) {
	    if (i === variable) return true;
	    i._inputs.forEach(inputs.add, inputs);
	  }
	  return false;
	}

	function variable_increment(variable) {
	  ++variable._indegree;
	}

	function variable_decrement(variable) {
	  --variable._indegree;
	}

	function variable_value(variable) {
	  return variable._promise.catch(variable._rejector);
	}

	function variable_invalidator(variable) {
	  return new Promise(function(resolve) {
	    variable._invalidate = resolve;
	  });
	}

	function variable_intersector(invalidation, variable) {
	  let node = typeof IntersectionObserver === "function" && variable._observer && variable._observer._node;
	  let visible = !node, resolve = noop, reject = noop, promise, observer;
	  if (node) {
	    observer = new IntersectionObserver(([entry]) => (visible = entry.isIntersecting) && (promise = null, resolve()));
	    observer.observe(node);
	    invalidation.then(() => (observer.disconnect(), observer = null, reject()));
	  }
	  return function(value) {
	    if (visible) return Promise.resolve(value);
	    if (!observer) return Promise.reject();
	    if (!promise) promise = new Promise((y, n) => (resolve = y, reject = n));
	    return promise.then(() => value);
	  };
	}

	function variable_compute(variable) {
	  variable._invalidate();
	  variable._invalidate = noop;
	  variable._pending();
	  var value0 = variable._value,
	      version = ++variable._version,
	      invalidation = null,
	      promise = variable._promise = Promise.all(variable._inputs.map(variable_value)).then(function(inputs) {
	    if (variable._version !== version) return;

	    // Replace any reference to invalidation with the promise, lazily.
	    for (var i = 0, n = inputs.length; i < n; ++i) {
	      switch (inputs[i]) {
	        case variable_invalidation: {
	          inputs[i] = invalidation = variable_invalidator(variable);
	          break;
	        }
	        case variable_visibility: {
	          if (!invalidation) invalidation = variable_invalidator(variable);
	          inputs[i] = variable_intersector(invalidation, variable);
	          break;
	        }
	      }
	    }

	    // Compute the initial value of the variable.
	    return variable._definition.apply(value0, inputs);
	  }).then(function(value) {
	    // If the value is a generator, then retrieve its first value,
	    // and dispose of the generator if the variable is invalidated.
	    // Note that the cell may already have been invalidated here,
	    // in which case we need to terminate the generator immediately!
	    if (generatorish(value)) {
	      if (variable._version !== version) return void value.return();
	      (invalidation || variable_invalidator(variable)).then(variable_return(value));
	      return variable_precompute(variable, version, promise, value);
	    }
	    return value;
	  });
	  promise.then(function(value) {
	    if (variable._version !== version) return;
	    variable._value = value;
	    variable._fulfilled(value);
	  }, function(error) {
	    if (variable._version !== version) return;
	    variable._value = undefined;
	    variable._rejected(error);
	  });
	}

	function variable_precompute(variable, version, promise, generator) {
	  function recompute() {
	    var promise = new Promise(function(resolve) {
	      resolve(generator.next());
	    }).then(function(next) {
	      return next.done ? undefined : Promise.resolve(next.value).then(function(value) {
	        if (variable._version !== version) return;
	        variable_postrecompute(variable, value, promise).then(recompute);
	        variable._fulfilled(value);
	        return value;
	      });
	    });
	    promise.catch(function(error) {
	      if (variable._version !== version) return;
	      variable_postrecompute(variable, undefined, promise);
	      variable._rejected(error);
	    });
	  }
	  return new Promise(function(resolve) {
	    resolve(generator.next());
	  }).then(function(next) {
	    if (next.done) return;
	    promise.then(recompute);
	    return next.value;
	  });
	}

	function variable_postrecompute(variable, value, promise) {
	  var runtime = variable._module._runtime;
	  variable._value = value;
	  variable._promise = promise;
	  variable._outputs.forEach(runtime._updates.add, runtime._updates); // TODO Cleaner?
	  return runtime._compute();
	}

	function variable_error(variable, error) {
	  variable._invalidate();
	  variable._invalidate = noop;
	  variable._pending();
	  ++variable._version;
	  variable._indegree = NaN;
	  (variable._promise = Promise.reject(error)).catch(noop);
	  variable._value = undefined;
	  variable._rejected(error);
	}

	function variable_return(generator) {
	  return function() {
	    generator.return();
	  };
	}

	function variable_reachable(variable) {
	  if (variable._observer !== no_observer) return true; // Directly reachable.
	  var outputs = new Set(variable._outputs);
	  for (const output of outputs) {
	    if (output._observer !== no_observer) return true;
	    output._outputs.forEach(outputs.add, outputs);
	  }
	  return false;
	}

	function window_global(name) {
	  return window[name];
	}

	function renderHtml(string) {
	  const template = document.createElement("template");
	  template.innerHTML = string;
	  return document.importNode(template.content, true);
	}

	function renderSvg(string) {
	  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
	  g.innerHTML = string;
	  return g;
	}

	const html = Object.assign(hypertext(renderHtml, fragment => {
	  if (fragment.firstChild === null) return null;
	  if (fragment.firstChild === fragment.lastChild) return fragment.removeChild(fragment.firstChild);
	  const span = document.createElement("span");
	  span.appendChild(fragment);
	  return span;
	}), {fragment: hypertext(renderHtml, fragment => fragment)});

	Object.assign(hypertext(renderSvg, g => {
	  if (g.firstChild === null) return null;
	  if (g.firstChild === g.lastChild) return g.removeChild(g.firstChild);
	  return g;
	}), {fragment: hypertext(renderSvg, g => {
	  const fragment = document.createDocumentFragment();
	  while (g.firstChild) fragment.appendChild(g.firstChild);
	  return fragment;
	})});

	const
	CODE_TAB = 9,
	CODE_LF = 10,
	CODE_FF = 12,
	CODE_CR = 13,
	CODE_SPACE = 32,
	CODE_UPPER_A = 65,
	CODE_UPPER_Z = 90,
	CODE_LOWER_A = 97,
	CODE_LOWER_Z = 122,
	CODE_LT = 60,
	CODE_GT = 62,
	CODE_SLASH = 47,
	CODE_DASH = 45,
	CODE_BANG = 33,
	CODE_EQ = 61,
	CODE_DQUOTE = 34,
	CODE_SQUOTE = 39,
	CODE_QUESTION = 63,
	STATE_DATA = 1,
	STATE_TAG_OPEN = 2,
	STATE_END_TAG_OPEN = 3,
	STATE_TAG_NAME = 4,
	STATE_BOGUS_COMMENT = 5,
	STATE_BEFORE_ATTRIBUTE_NAME = 6,
	STATE_AFTER_ATTRIBUTE_NAME = 7,
	STATE_ATTRIBUTE_NAME = 8,
	STATE_BEFORE_ATTRIBUTE_VALUE = 9,
	STATE_ATTRIBUTE_VALUE_DOUBLE_QUOTED = 10,
	STATE_ATTRIBUTE_VALUE_SINGLE_QUOTED = 11,
	STATE_ATTRIBUTE_VALUE_UNQUOTED = 12,
	STATE_AFTER_ATTRIBUTE_VALUE_QUOTED = 13,
	STATE_SELF_CLOSING_START_TAG = 14,
	STATE_COMMENT_START = 15,
	STATE_COMMENT_START_DASH = 16,
	STATE_COMMENT = 17,
	STATE_COMMENT_LESS_THAN_SIGN = 18,
	STATE_COMMENT_LESS_THAN_SIGN_BANG = 19,
	STATE_COMMENT_LESS_THAN_SIGN_BANG_DASH = 20,
	STATE_COMMENT_LESS_THAN_SIGN_BANG_DASH_DASH = 21,
	STATE_COMMENT_END_DASH = 22,
	STATE_COMMENT_END = 23,
	STATE_COMMENT_END_BANG = 24,
	STATE_MARKUP_DECLARATION_OPEN = 25,
	STATE_RAWTEXT = 26,
	STATE_RAWTEXT_LESS_THAN_SIGN = 27,
	STATE_RAWTEXT_END_TAG_OPEN = 28,
	STATE_RAWTEXT_END_TAG_NAME = 29,
	SHOW_COMMENT = 128,
	SHOW_ELEMENT = 1,
	TYPE_COMMENT = 8,
	TYPE_ELEMENT = 1,
	NS_SVG = "http://www.w3.org/2000/svg",
	NS_XLINK = "http://www.w3.org/1999/xlink",
	NS_XML = "http://www.w3.org/XML/1998/namespace",
	NS_XMLNS = "http://www.w3.org/2000/xmlns/";

	const svgAdjustAttributes = new Map([
	  "attributeName",
	  "attributeType",
	  "baseFrequency",
	  "baseProfile",
	  "calcMode",
	  "clipPathUnits",
	  "diffuseConstant",
	  "edgeMode",
	  "filterUnits",
	  "glyphRef",
	  "gradientTransform",
	  "gradientUnits",
	  "kernelMatrix",
	  "kernelUnitLength",
	  "keyPoints",
	  "keySplines",
	  "keyTimes",
	  "lengthAdjust",
	  "limitingConeAngle",
	  "markerHeight",
	  "markerUnits",
	  "markerWidth",
	  "maskContentUnits",
	  "maskUnits",
	  "numOctaves",
	  "pathLength",
	  "patternContentUnits",
	  "patternTransform",
	  "patternUnits",
	  "pointsAtX",
	  "pointsAtY",
	  "pointsAtZ",
	  "preserveAlpha",
	  "preserveAspectRatio",
	  "primitiveUnits",
	  "refX",
	  "refY",
	  "repeatCount",
	  "repeatDur",
	  "requiredExtensions",
	  "requiredFeatures",
	  "specularConstant",
	  "specularExponent",
	  "spreadMethod",
	  "startOffset",
	  "stdDeviation",
	  "stitchTiles",
	  "surfaceScale",
	  "systemLanguage",
	  "tableValues",
	  "targetX",
	  "targetY",
	  "textLength",
	  "viewBox",
	  "viewTarget",
	  "xChannelSelector",
	  "yChannelSelector",
	  "zoomAndPan"
	].map(name => [name.toLowerCase(), name]));

	const svgForeignAttributes = new Map([
	  ["xlink:actuate", NS_XLINK],
	  ["xlink:arcrole", NS_XLINK],
	  ["xlink:href", NS_XLINK],
	  ["xlink:role", NS_XLINK],
	  ["xlink:show", NS_XLINK],
	  ["xlink:title", NS_XLINK],
	  ["xlink:type", NS_XLINK],
	  ["xml:lang", NS_XML],
	  ["xml:space", NS_XML],
	  ["xmlns", NS_XMLNS],
	  ["xmlns:xlink", NS_XMLNS]
	]);

	function hypertext(render, postprocess) {
	  return function({raw: strings}) {
	    let state = STATE_DATA;
	    let string = "";
	    let tagNameStart; // either an open tag or an end tag
	    let tagName; // only open; beware nesting! used only for rawtext
	    let attributeNameStart;
	    let attributeNameEnd;
	    let nodeFilter = 0;

	    for (let j = 0, m = arguments.length; j < m; ++j) {
	      const input = strings[j];

	      if (j > 0) {
	        const value = arguments[j];
	        switch (state) {
	          case STATE_RAWTEXT: {
	            if (value != null) {
	              const text = `${value}`;
	              if (isEscapableRawText(tagName)) {
	                string += text.replace(/[<]/g, entity);
	              } else if (new RegExp(`</${tagName}[\\s>/]`, "i").test(string.slice(-tagName.length - 2) + text)) {
	                throw new Error("unsafe raw text"); // appropriate end tag
	              } else {
	                string += text;
	              }
	            }
	            break;
	          }
	          case STATE_DATA: {
	            if (value == null) ; else if (value instanceof Node
	                || (typeof value !== "string" && value[Symbol.iterator])
	                || (/(?:^|>)$/.test(strings[j - 1]) && /^(?:<|$)/.test(input))) {
	              string += "<!--::" + j + "-->";
	              nodeFilter |= SHOW_COMMENT;
	            } else {
	              string += `${value}`.replace(/[<&]/g, entity);
	            }
	            break;
	          }
	          case STATE_BEFORE_ATTRIBUTE_VALUE: {
	            state = STATE_ATTRIBUTE_VALUE_UNQUOTED;
	            let text;
	            if (/^[\s>]/.test(input)) {
	              if (value == null || value === false) {
	                string = string.slice(0, attributeNameStart - strings[j - 1].length);
	                break;
	              }
	              if (value === true || (text = `${value}`) === "") {
	                string += "''";
	                break;
	              }
	              const name = strings[j - 1].slice(attributeNameStart, attributeNameEnd);
	              if ((name === "style" && isObjectLiteral(value)) || typeof value === "function") {
	                string += "::" + j;
	                nodeFilter |= SHOW_ELEMENT;
	                break;
	              }
	            }
	            if (text === undefined) text = `${value}`;
	            if (text === "") throw new Error("unsafe unquoted empty string");
	            string += text.replace(/^['"]|[\s>&]/g, entity);
	            break;
	          }
	          case STATE_ATTRIBUTE_VALUE_UNQUOTED: {
	            string += `${value}`.replace(/[\s>&]/g, entity);
	            break;
	          }
	          case STATE_ATTRIBUTE_VALUE_SINGLE_QUOTED: {
	            string += `${value}`.replace(/['&]/g, entity);
	            break;
	          }
	          case STATE_ATTRIBUTE_VALUE_DOUBLE_QUOTED: {
	            string += `${value}`.replace(/["&]/g, entity);
	            break;
	          }
	          case STATE_BEFORE_ATTRIBUTE_NAME: {
	            if (isObjectLiteral(value)) {
	              string += "::" + j + "=''";
	              nodeFilter |= SHOW_ELEMENT;
	              break;
	            }
	            throw new Error("invalid binding");
	          }
	          case STATE_COMMENT: break;
	          default: throw new Error("invalid binding");
	        }
	      }

	      for (let i = 0, n = input.length; i < n; ++i) {
	        const code = input.charCodeAt(i);

	        switch (state) {
	          case STATE_DATA: {
	            if (code === CODE_LT) {
	              state = STATE_TAG_OPEN;
	            }
	            break;
	          }
	          case STATE_TAG_OPEN: {
	            if (code === CODE_BANG) {
	              state = STATE_MARKUP_DECLARATION_OPEN;
	            } else if (code === CODE_SLASH) {
	              state = STATE_END_TAG_OPEN;
	            } else if (isAsciiAlphaCode(code)) {
	              tagNameStart = i, tagName = undefined;
	              state = STATE_TAG_NAME, --i;
	            } else if (code === CODE_QUESTION) {
	              state = STATE_BOGUS_COMMENT, --i;
	            } else {
	              state = STATE_DATA, --i;
	            }
	            break;
	          }
	          case STATE_END_TAG_OPEN: {
	            if (isAsciiAlphaCode(code)) {
	              state = STATE_TAG_NAME, --i;
	            } else if (code === CODE_GT) {
	              state = STATE_DATA;
	            } else {
	              state = STATE_BOGUS_COMMENT, --i;
	            }
	            break;
	          }
	          case STATE_TAG_NAME: {
	            if (isSpaceCode(code)) {
	              state = STATE_BEFORE_ATTRIBUTE_NAME;
	              tagName = lower(input, tagNameStart, i);
	            } else if (code === CODE_SLASH) {
	              state = STATE_SELF_CLOSING_START_TAG;
	            } else if (code === CODE_GT) {
	              tagName = lower(input, tagNameStart, i);
	              state = isRawText(tagName) ? STATE_RAWTEXT : STATE_DATA;
	            }
	            break;
	          }
	          case STATE_BEFORE_ATTRIBUTE_NAME: {
	            if (isSpaceCode(code)) ; else if (code === CODE_SLASH || code === CODE_GT) {
	              state = STATE_AFTER_ATTRIBUTE_NAME, --i;
	            } else if (code === CODE_EQ) {
	              state = STATE_ATTRIBUTE_NAME;
	              attributeNameStart = i + 1, attributeNameEnd = undefined;
	            } else {
	              state = STATE_ATTRIBUTE_NAME, --i;
	              attributeNameStart = i + 1, attributeNameEnd = undefined;
	            }
	            break;
	          }
	          case STATE_ATTRIBUTE_NAME: {
	            if (isSpaceCode(code) || code === CODE_SLASH || code === CODE_GT) {
	              state = STATE_AFTER_ATTRIBUTE_NAME, --i;
	              attributeNameEnd = i;
	            } else if (code === CODE_EQ) {
	              state = STATE_BEFORE_ATTRIBUTE_VALUE;
	              attributeNameEnd = i;
	            }
	            break;
	          }
	          case STATE_AFTER_ATTRIBUTE_NAME: {
	            if (isSpaceCode(code)) ; else if (code === CODE_SLASH) {
	              state = STATE_SELF_CLOSING_START_TAG;
	            } else if (code === CODE_EQ) {
	              state = STATE_BEFORE_ATTRIBUTE_VALUE;
	            } else if (code === CODE_GT) {
	              state = isRawText(tagName) ? STATE_RAWTEXT : STATE_DATA;
	            } else {
	              state = STATE_ATTRIBUTE_NAME, --i;
	              attributeNameStart = i + 1, attributeNameEnd = undefined;
	            }
	            break;
	          }
	          case STATE_BEFORE_ATTRIBUTE_VALUE: {
	            if (isSpaceCode(code)) ; else if (code === CODE_DQUOTE) {
	              state = STATE_ATTRIBUTE_VALUE_DOUBLE_QUOTED;
	            } else if (code === CODE_SQUOTE) {
	              state = STATE_ATTRIBUTE_VALUE_SINGLE_QUOTED;
	            } else if (code === CODE_GT) {
	              state = isRawText(tagName) ? STATE_RAWTEXT : STATE_DATA;
	            } else {
	              state = STATE_ATTRIBUTE_VALUE_UNQUOTED, --i;
	            }
	            break;
	          }
	          case STATE_ATTRIBUTE_VALUE_DOUBLE_QUOTED: {
	            if (code === CODE_DQUOTE) {
	              state = STATE_AFTER_ATTRIBUTE_VALUE_QUOTED;
	            }
	            break;
	          }
	          case STATE_ATTRIBUTE_VALUE_SINGLE_QUOTED: {
	            if (code === CODE_SQUOTE) {
	              state = STATE_AFTER_ATTRIBUTE_VALUE_QUOTED;
	            }
	            break;
	          }
	          case STATE_ATTRIBUTE_VALUE_UNQUOTED: {
	            if (isSpaceCode(code)) {
	              state = STATE_BEFORE_ATTRIBUTE_NAME;
	            } else if (code === CODE_GT) {
	              state = isRawText(tagName) ? STATE_RAWTEXT : STATE_DATA;
	            }
	            break;
	          }
	          case STATE_AFTER_ATTRIBUTE_VALUE_QUOTED: {
	            if (isSpaceCode(code)) {
	              state = STATE_BEFORE_ATTRIBUTE_NAME;
	            } else if (code === CODE_SLASH) {
	              state = STATE_SELF_CLOSING_START_TAG;
	            } else if (code === CODE_GT) {
	              state = isRawText(tagName) ? STATE_RAWTEXT : STATE_DATA;
	            } else {
	              state = STATE_BEFORE_ATTRIBUTE_NAME, --i;
	            }
	            break;
	          }
	          case STATE_SELF_CLOSING_START_TAG: {
	            if (code === CODE_GT) {
	              state = STATE_DATA;
	            } else {
	              state = STATE_BEFORE_ATTRIBUTE_NAME, --i;
	            }
	            break;
	          }
	          case STATE_BOGUS_COMMENT: {
	            if (code === CODE_GT) {
	              state = STATE_DATA;
	            }
	            break;
	          }
	          case STATE_COMMENT_START: {
	            if (code === CODE_DASH) {
	              state = STATE_COMMENT_START_DASH;
	            } else if (code === CODE_GT) {
	              state = STATE_DATA;
	            } else {
	              state = STATE_COMMENT, --i;
	            }
	            break;
	          }
	          case STATE_COMMENT_START_DASH: {
	            if (code === CODE_DASH) {
	              state = STATE_COMMENT_END;
	            } else if (code === CODE_GT) {
	              state = STATE_DATA;
	            } else {
	              state = STATE_COMMENT, --i;
	            }
	            break;
	          }
	          case STATE_COMMENT: {
	            if (code === CODE_LT) {
	              state = STATE_COMMENT_LESS_THAN_SIGN;
	            } else if (code === CODE_DASH) {
	              state = STATE_COMMENT_END_DASH;
	            }
	            break;
	          }
	          case STATE_COMMENT_LESS_THAN_SIGN: {
	            if (code === CODE_BANG) {
	              state = STATE_COMMENT_LESS_THAN_SIGN_BANG;
	            } else if (code !== CODE_LT) {
	              state = STATE_COMMENT, --i;
	            }
	            break;
	          }
	          case STATE_COMMENT_LESS_THAN_SIGN_BANG: {
	            if (code === CODE_DASH) {
	              state = STATE_COMMENT_LESS_THAN_SIGN_BANG_DASH;
	            } else {
	              state = STATE_COMMENT, --i;
	            }
	            break;
	          }
	          case STATE_COMMENT_LESS_THAN_SIGN_BANG_DASH: {
	            if (code === CODE_DASH) {
	              state = STATE_COMMENT_LESS_THAN_SIGN_BANG_DASH_DASH;
	            } else {
	              state = STATE_COMMENT_END, --i;
	            }
	            break;
	          }
	          case STATE_COMMENT_LESS_THAN_SIGN_BANG_DASH_DASH: {
	            state = STATE_COMMENT_END, --i;
	            break;
	          }
	          case STATE_COMMENT_END_DASH: {
	            if (code === CODE_DASH) {
	              state = STATE_COMMENT_END;
	            } else {
	              state = STATE_COMMENT, --i;
	            }
	            break;
	          }
	          case STATE_COMMENT_END: {
	            if (code === CODE_GT) {
	              state = STATE_DATA;
	            } else if (code === CODE_BANG) {
	              state = STATE_COMMENT_END_BANG;
	            } else if (code !== CODE_DASH) {
	              state = STATE_COMMENT, --i;
	            }
	            break;
	          }
	          case STATE_COMMENT_END_BANG: {
	            if (code === CODE_DASH) {
	              state = STATE_COMMENT_END_DASH;
	            } else if (code === CODE_GT) {
	              state = STATE_DATA;
	            } else {
	              state = STATE_COMMENT, --i;
	            }
	            break;
	          }
	          case STATE_MARKUP_DECLARATION_OPEN: {
	            if (code === CODE_DASH && input.charCodeAt(i + 1) === CODE_DASH) {
	              state = STATE_COMMENT_START, ++i;
	            } else { // Note: CDATA and DOCTYPE unsupported!
	              state = STATE_BOGUS_COMMENT, --i;
	            }
	            break;
	          }
	          case STATE_RAWTEXT: {
	            if (code === CODE_LT) {
	              state = STATE_RAWTEXT_LESS_THAN_SIGN;
	            }
	            break;
	          }
	          case STATE_RAWTEXT_LESS_THAN_SIGN: {
	            if (code === CODE_SLASH) {
	              state = STATE_RAWTEXT_END_TAG_OPEN;
	            } else {
	              state = STATE_RAWTEXT, --i;
	            }
	            break;
	          }
	          case STATE_RAWTEXT_END_TAG_OPEN: {
	            if (isAsciiAlphaCode(code)) {
	              tagNameStart = i;
	              state = STATE_RAWTEXT_END_TAG_NAME, --i;
	            } else {
	              state = STATE_RAWTEXT, --i;
	            }
	            break;
	          }
	          case STATE_RAWTEXT_END_TAG_NAME: {
	            if (isSpaceCode(code) && tagName === lower(input, tagNameStart, i)) {
	              state = STATE_BEFORE_ATTRIBUTE_NAME;
	            } else if (code === CODE_SLASH && tagName === lower(input, tagNameStart, i)) {
	              state = STATE_SELF_CLOSING_START_TAG;
	            } else if (code === CODE_GT && tagName === lower(input, tagNameStart, i)) {
	              state = STATE_DATA;
	            } else if (!isAsciiAlphaCode(code)) {
	              state = STATE_RAWTEXT, --i;
	            }
	            break;
	          }
	          default: {
	            state = undefined;
	            break;
	          }
	        }
	      }

	      string += input;
	    }

	    const root = render(string);

	    const walker = document.createTreeWalker(root, nodeFilter, null, false);
	    const removeNodes = [];
	    while (walker.nextNode()) {
	      const node = walker.currentNode;
	      switch (node.nodeType) {
	        case TYPE_ELEMENT: {
	          const attributes = node.attributes;
	          for (let i = 0, n = attributes.length; i < n; ++i) {
	            const {name, value: currentValue} = attributes[i];
	            if (/^::/.test(name)) {
	              const value = arguments[+name.slice(2)];
	              removeAttribute(node, name), --i, --n;
	              for (const key in value) {
	                const subvalue = value[key];
	                if (subvalue == null || subvalue === false) ; else if (typeof subvalue === "function") {
	                  node[key] = subvalue;
	                } else if (key === "style" && isObjectLiteral(subvalue)) {
	                  setStyles(node[key], subvalue);
	                } else {
	                  setAttribute(node, key, subvalue === true ? "" : subvalue);
	                }
	              }
	            } else if (/^::/.test(currentValue)) {
	              const value = arguments[+currentValue.slice(2)];
	              removeAttribute(node, name), --i, --n;
	              if (typeof value === "function") {
	                node[name] = value;
	              } else { // style
	                setStyles(node[name], value);
	              }
	            }
	          }
	          break;
	        }
	        case TYPE_COMMENT: {
	          if (/^::/.test(node.data)) {
	            const parent = node.parentNode;
	            const value = arguments[+node.data.slice(2)];
	            if (value instanceof Node) {
	              parent.insertBefore(value, node);
	            } else if (typeof value !== "string" && value[Symbol.iterator]) {
	              if (value instanceof NodeList || value instanceof HTMLCollection) {
	                for (let i = value.length - 1, r = node; i >= 0; --i) {
	                  r = parent.insertBefore(value[i], r);
	                }
	              } else {
	                for (const subvalue of value) {
	                  if (subvalue == null) continue;
	                  parent.insertBefore(subvalue instanceof Node ? subvalue : document.createTextNode(subvalue), node);
	                }
	              }
	            } else {
	              parent.insertBefore(document.createTextNode(value), node);
	            }
	            removeNodes.push(node);
	          }
	          break;
	        }
	      }
	    }

	    for (const node of removeNodes) {
	      node.parentNode.removeChild(node);
	    }

	    return postprocess(root);
	  };
	}

	function entity(character) {
	  return `&#${character.charCodeAt(0).toString()};`;
	}

	function isAsciiAlphaCode(code) {
	  return (CODE_UPPER_A <= code && code <= CODE_UPPER_Z)
	      || (CODE_LOWER_A <= code && code <= CODE_LOWER_Z);
	}

	function isSpaceCode(code) {
	  return code === CODE_TAB
	      || code === CODE_LF
	      || code === CODE_FF
	      || code === CODE_SPACE
	      || code === CODE_CR; // normalize newlines
	}

	function isObjectLiteral(value) {
	  return value && value.toString === Object.prototype.toString;
	}

	function isRawText(tagName) {
	  return tagName === "script" || tagName === "style" || isEscapableRawText(tagName);
	}

	function isEscapableRawText(tagName) {
	  return tagName === "textarea" || tagName === "title";
	}

	function lower(input, start, end) {
	  return input.slice(start, end).toLowerCase();
	}

	function setAttribute(node, name, value) {
	  if (node.namespaceURI === NS_SVG) {
	    name = name.toLowerCase();
	    name = svgAdjustAttributes.get(name) || name;
	    if (svgForeignAttributes.has(name)) {
	      node.setAttributeNS(svgForeignAttributes.get(name), name, value);
	      return;
	    }
	  }
	  node.setAttribute(name, value);
	}

	function removeAttribute(node, name) {
	  if (node.namespaceURI === NS_SVG) {
	    name = name.toLowerCase();
	    name = svgAdjustAttributes.get(name) || name;
	    if (svgForeignAttributes.has(name)) {
	      node.removeAttributeNS(svgForeignAttributes.get(name), name);
	      return;
	    }
	  }
	  node.removeAttribute(name);
	}

	// We can’t use Object.assign because custom properties…
	function setStyles(style, values) {
	  for (const name in values) {
	    const value = values[name];
	    if (name.startsWith("--")) style.setProperty(name, value);
	    else style[name] = value;
	  }
	}

	function length(x) {
	  return x == null ? null : typeof x === "number" ? `${x}px` : `${x}`;
	}

	function maybeWidth(width) {
	  return {"--input-width": length(width)};
	}

	const bubbles = {bubbles: true};

	function preventDefault(event) {
	  event.preventDefault();
	}

	function dispatchInput({currentTarget}) {
	  (currentTarget.form || currentTarget).dispatchEvent(new Event("input", bubbles));
	}

	function checkValidity(input) {
	  return input.checkValidity();
	}

	function identity(x) {
	  return x;
	}

	let nextId = 0;

	function newId() {
	  return `__ns__-${++nextId}`;
	}

	function maybeLabel(label, input) {
	  if (!label) return;
	  label = html`<label>${label}`;
	  if (input !== undefined) label.htmlFor = input.id = newId();
	  return label;
	}

	function arrayify(array) {
	  return Array.isArray(array) ? array : Array.from(array);
	}

	// Note: use formatAuto (or any other localized format) to present values to the
	// user; stringify is only intended for machine values.
	function stringify(x) {
	  return x == null ? "" : `${x}`;
	}

	const formatLocaleAuto = localize(locale => {
	  const formatNumber = formatLocaleNumber(locale);
	  return value => value == null ? ""
	    : typeof value === "number" ? formatNumber(value)
	    : value instanceof Date ? formatDate(value)
	    : `${value}`;
	});

	const formatLocaleNumber = localize(locale => {
	  return value => value === 0 ? "0" : value.toLocaleString(locale); // handle negative zero
	});

	formatLocaleAuto();

	formatLocaleNumber();

	function formatTrim(value) {
	  const s = value.toString();
	  const n = s.length;
	  let i0 = -1, i1;
	  out: for (let i = 1; i < n; ++i) {
	    switch (s[i]) {
	      case ".": i0 = i1 = i; break;
	      case "0": if (i0 === 0) i0 = i; i1 = i; break;
	      default: if (!+s[i]) break out; if (i0 > 0) i0 = 0; break;
	    }
	  }
	  return i0 > 0 ? s.slice(0, i0) + s.slice(i1 + 1) : s;
	}

	function formatDate(date) {
	  return format(date, "Invalid Date");
	}

	// Memoize the last-returned locale.
	function localize(f) {
	  let key = localize, value;
	  return (locale = "en") => locale === key ? value : (value = f(key = locale));
	}

	function ascending(a, b) {
	  return defined(b) - defined(a) || (a < b ? -1 : a > b ? 1 : a >= b ? 0 : NaN);
	}

	function descending(b, a) {
	  return defined(a) - defined(b) || (a < b ? -1 : a > b ? 1 : a >= b ? 0 : NaN);
	}

	function defined(d) {
	  return d != null && !Number.isNaN(d);
	}

	const first = ([x]) => x;
	const second = ([, x]) => x;

	function createChooser({multiple: fixedMultiple, render, selectedIndexes, select}) {
	  return function chooser(data, {
	    locale,
	    keyof = data instanceof Map ? first : identity,
	    valueof = data instanceof Map ? second : identity,
	    format = data instanceof Map ? first : formatLocaleAuto(locale),
	    multiple,
	    key,
	    value,
	    disabled = false,
	    sort,
	    unique,
	    ...options
	  } = {}) {
	    if (typeof keyof !== "function") throw new TypeError("keyof is not a function");
	    if (typeof valueof !== "function") throw new TypeError("valueof is not a function");
	    if (typeof format !== "function") throw new TypeError("format is not a function");
	    if (fixedMultiple !== undefined) multiple = fixedMultiple;
	    sort = maybeSort(sort);
	    let size = +multiple;
	    if (value === undefined) value = key !== undefined && data instanceof Map ? (size > 0 ? Array.from(key, k => data.get(k)) : data.get(key)) : undefined;
	    unique = !!unique;
	    data = arrayify(data);
	    let keys = data.map((d, i) => [keyof(d, i, data), i]);
	    if (sort !== undefined) keys.sort(([a], [b]) => sort(a, b));
	    if (unique) keys = [...new Map(keys.map(o => [intern(o[0]), o])).values()];
	    const index = keys.map(second);
	    if (multiple === true) size = Math.max(1, Math.min(10, index.length));
	    else if (size > 0) multiple = true;
	    else multiple = false, size = undefined;
	    const [form, input] = render(
	      data,
	      index,
	      maybeSelection(data, index, value, multiple, valueof),
	      maybeDisabled(data, index, disabled, valueof),
	      {
	        ...options,
	        format,
	        multiple,
	        size
	      }
	    );
	    form.onchange = dispatchInput;
	    form.oninput = oninput;
	    form.onsubmit = preventDefault;
	    function oninput(event) {
	      if (event && event.isTrusted) form.onchange = null;
	      if (multiple) {
	        value = selectedIndexes(input).map(i => valueof(data[i], i, data));
	      } else {
	        const i = selectedIndex(input);
	        value = i < 0 ? null : valueof(data[i], i, data);
	      }
	    }
	    oninput();
	    return Object.defineProperty(form, "value", {
	      get() {
	        return value;
	      },
	      set(v) {
	        if (multiple) {
	          const selection = new Set(v);
	          for (const e of input) {
	            const i = +e.value;
	            select(e, selection.has(valueof(data[i], i, data)));
	          }
	        } else {
	          input.value = index.find(i => v === valueof(data[i], i, data));
	        }
	        oninput();
	      }
	    });
	  };
	}

	function maybeSelection(data, index, value, multiple, valueof) {
	  const values = new Set(value === undefined ? [] : multiple ? arrayify(value) : [value]);
	  if (!values.size) return () => false;
	  const selection = new Set();
	  for (const i of index) {
	    if (values.has(valueof(data[i], i, data))) {
	      selection.add(i);
	    }
	  }
	  return i => selection.has(i);
	}

	function maybeDisabled(data, index, value, valueof) {
	  if (typeof value === "boolean") return value;
	  const values = new Set(arrayify(value));
	  const disabled = new Set();
	  for (const i of index) {
	    if (values.has(valueof(data[i], i, data))) {
	      disabled.add(i);
	    }
	  }
	  return i => disabled.has(i);
	}

	function maybeSort(sort) {
	  if (sort === undefined || sort === false) return;
	  if (sort === true || sort === "ascending") return ascending;
	  if (sort === "descending") return descending;
	  if (typeof sort === "function") return sort;
	  throw new TypeError("sort is not a function");
	}

	function selectedIndex(input) {
	  return input.value ? +input.value : -1;
	}

	function intern(value) {
	  return value !== null && typeof value === "object" ? value.valueOf() : value;
	}

	function createCheckbox(multiple, type) {
	  return createChooser({
	    multiple,
	    render(data, index, selected, disabled, {format, label}) {
	      const form = html`<form class="__ns__ __ns__-checkbox">
      ${maybeLabel(label)}<div>
        ${index.map(i => html`<label><input type=${type} disabled=${typeof disabled === "function" ? disabled(i) : disabled} name=input value=${i} checked=${selected(i)}>${format(data[i], i, data)}`)}
      </div>
    </form>`;
	      return [form, inputof$1(form.elements.input, multiple)];
	    },
	    selectedIndexes(input) {
	      return Array.from(input).filter(i => i.checked).map(i => +i.value);
	    },
	    select(input, selected) {
	      input.checked = selected;
	    }
	  });
	}

	const checkbox = createCheckbox(true, "checkbox");

	// The input is undefined if there are no options, or an individual input
	// element if there is only one; we want these two cases to behave the same as
	// when there are two or more options, i.e., a RadioNodeList.
	function inputof$1(input, multiple) {
	  return input === undefined ? new OptionZero(multiple ? [] : null)
	    : typeof input.length === "undefined" ? new (multiple ? MultipleOptionOne : OptionOne)(input)
	    : input;
	}

	class OptionZero {
	  constructor(value) {
	    this._value = value;
	  }
	  get value() {
	    return this._value;
	  }
	  set value(v) {
	    // ignore
	  }
	  *[Symbol.iterator]() {
	    // empty
	  }
	}

	// TODO If we allow selected radios to be cleared by command-clicking, then
	// assigning a radio’s value programmatically should also clear the selection.
	// This will require changing this class and also wrapping RadioNodeList in the
	// common case to change the value setter’s behavior.
	class OptionOne {
	  constructor(input) {
	    this._input = input;
	  }
	  get value() {
	    const {_input} = this;
	    return _input.checked ? _input.value : "";
	  }
	  set value(v) {
	    const {_input} = this;
	    if (_input.checked) return;
	    _input.checked = stringify(v) === _input.value;
	  }
	  *[Symbol.iterator]() {
	    yield this._input;
	  }
	}

	class MultipleOptionOne {
	  constructor(input) {
	    this._input = input;
	    this._value = input.checked ? [input.value] : [];
	  }
	  get value() {
	    return this._value;
	  }
	  set value(v) {
	    const {_input} = this;
	    if (_input.checked) return;
	    _input.checked = stringify(v) === _input.value;
	    this._value = _input.checked ? [_input.value] : [];
	  }
	  *[Symbol.iterator]() {
	    yield this._input;
	  }
	}

	const epsilon = 1e-6;

	function range(extent = [0, 1], options) {
	  return createRange({extent, range: true}, options);
	}

	function createRange({
	  extent: [min, max],
	  range
	}, {
	  format = formatTrim,
	  transform,
	  invert,
	  label = "",
	  value: initialValue,
	  step,
	  disabled,
	  placeholder,
	  validate = checkValidity,
	  width
	} = {}) {
	  let value;
	  if (typeof format !== "function") throw new TypeError("format is not a function");
	  if (min == null || isNaN(min = +min)) min = -Infinity;
	  if (max == null || isNaN(max = +max)) max = Infinity;
	  if (min > max) [min, max] = [max, min], transform === undefined && (transform = negate);
	  if (step !== undefined) step = +step;
	  const number = html`<input type=number min=${isFinite(min) ? min : null} max=${isFinite(max) ? max : null} step=${step == undefined ? "any" : step} name=number required placeholder=${placeholder} oninput=${onnumber} disabled=${disabled}>`;
	  let irange; // untransformed range for coercion
	  if (range) {
	    if (transform === undefined) transform = identity;
	    if (typeof transform !== "function") throw new TypeError("transform is not a function");
	    if (invert === undefined) invert = transform.invert === undefined ? solver(transform) : transform.invert;
	    if (typeof invert !== "function") throw new TypeError("invert is not a function");
	    let tmin = +transform(min), tmax = +transform(max);
	    if (tmin > tmax) [tmin, tmax] = [tmax, tmin];
	    range = html`<input type=range min=${isFinite(tmin) ? tmin : null} max=${isFinite(tmax) ? tmax : null} step=${step === undefined || (transform !== identity && transform !== negate) ? "any" : step} name=range oninput=${onrange} disabled=${disabled}>`;
	    irange = transform === identity ? range : html`<input type=range min=${min} max=${max} step=${step === undefined ? "any" : step} name=range disabled=${disabled}>`;
	  } else {
	    range = null;
	    transform = invert = identity;
	  }
	  const form = html`<form class=__ns__ onsubmit=${preventDefault} style=${maybeWidth(width)}>
    ${maybeLabel(label, number)}<div class=__ns__-input>
      ${number}${range}
    </div>
  </form>`;
	  // If range, use an untransformed range to round to the nearest valid value.
	  function coerce(v) {
	    if (!irange) return +v;
	    v = Math.max(min, Math.min(max, v));
	    if (!isFinite(v)) return v;
	    irange.valueAsNumber = v;
	    return irange.valueAsNumber;
	  }
	  function onrange(event) {
	    const v = coerce(invert(range.valueAsNumber));
	    if (isFinite(v)) {
	      number.valueAsNumber = Math.max(min, Math.min(max, v));
	      if (validate(number)) {
	        value = number.valueAsNumber;
	        number.value = format(value);
	        return;
	      }
	    }
	    if (event) event.stopPropagation();
	  }
	  function onnumber(event) {
	    const v = coerce(number.valueAsNumber);
	    if (isFinite(v)) {
	      if (range) range.valueAsNumber = transform(v);
	      if (validate(number)) {
	        value = v;
	        return;
	      }
	    }
	    if (event) event.stopPropagation();
	  }
	  Object.defineProperty(form, "value", {
	    get() {
	      return value;
	    },
	    set(v) {
	      v = coerce(v);
	      if (isFinite(v)) {
	        number.valueAsNumber = v;
	        if (range) range.valueAsNumber = transform(v);
	        if (validate(number)) {
	          value = v;
	          number.value = format(value);
	        }
	      }
	    }
	  });
	  if (initialValue === undefined && irange) initialValue = irange.valueAsNumber; // (min + max) / 2
	  if (initialValue !== undefined) form.value = initialValue; // invoke setter
	  return form;
	}

	function negate(x) {
	  return -x;
	}

	function square(x) {
	  return x * x;
	}

	function solver(f) {
	  if (f === identity || f === negate) return f;
	  if (f === Math.sqrt) return square;
	  if (f === Math.log) return Math.exp;
	  if (f === Math.exp) return Math.log;
	  return x => solve(f, x, x);
	}

	function solve(f, y, x) {
	  let steps = 100, delta, f0, f1;
	  x = x === undefined ? 0 : +x;
	  y = +y;
	  do {
	    f0 = f(x);
	    f1 = f(x + epsilon);
	    if (f0 === f1) f1 = f0 + epsilon;
	    x -= delta = (-1 * epsilon * (f0 - y)) / (f0 - f1);
	  } while (steps-- > 0 && Math.abs(delta) > epsilon);
	  return steps < 0 ? NaN : x;
	}

	const select = createChooser({
	  render(data, index, selected, disabled, {format, multiple, size, label, width}) {
	    const select = html`<select class=__ns__-input disabled=${disabled === true} multiple=${multiple} size=${size} name=input>
      ${index.map(i => html`<option value=${i} disabled=${typeof disabled === "function" ? disabled(i) : false} selected=${selected(i)}>${stringify(format(data[i], i, data))}`)}
    </select>`;
	    const form = html`<form class=__ns__ style=${maybeWidth(width)}>${maybeLabel(label, select)}${select}`;
	    return [form, select];
	  },
	  selectedIndexes(input) {
	    return Array.from(input.selectedOptions, i => +i.value);
	  },
	  select(input, selected) {
	    input.selected = selected;
	  }
	});

	const rowHeight = 22;

	function table$1(data, options = {}) {
	  const {
	    rows = 11.5, // maximum number of rows to show
	    height,
	    maxHeight = height === undefined ? (rows + 1) * rowHeight - 1 : undefined,
	    width = {}, // object of column name to width, or overall table width
	    maxWidth
	  } = options;
	  const id = newId();
	  const root = html`<form class="__ns__ __ns__-table" id=${id} style=${{height: length(height), maxHeight: length(maxHeight), width: typeof width === "string" || typeof width === "number" ? length(width) : undefined, maxWidth: length(maxWidth)}}>`;
	  // The outer form element is created synchronously, while the table is lazily
	  // created when the data promise resolves. This allows you to pass a promise
	  // of data to the table without an explicit await.
	  if (data && typeof data.then === "function") {
	    Object.defineProperty(root, "value", {
	      set() {
	        throw new Error("cannot set value while data is unresolved");
	      }
	    });
	    Promise.resolve(data).then(data => initialize({root, id}, data, options));
	  } else {
	    initialize({root, id}, data, options);
	  }
	  return root;
	}

	function initialize(
	  {
	    root,
	    id
	  },
	  data,
	  {
	    columns, // array of column names
	    value, // initial selection
	    required = true, // if true, the value is everything if nothing is selected
	    sort, // name of column to sort by, if any
	    reverse = false, // if sorting, true for descending and false for ascending
	    format, // object of column name to format function
	    locale,
	    align, // object of column name to left, right, or center
	    header, // object of column name to string or HTML element
	    rows = 11.5, // maximum number of rows to show
	    width = {}, // object of column name to width, or overall table width
	    multiple = true,
	    layout // "fixed" or "auto"
	  } = {}
	) {
	  columns = columns === undefined ? columnsof(data) : arrayify(columns);
	  if (layout === undefined) layout = columns.length >= 12 ? "auto" : "fixed";
	  format = formatof(format, data, columns, locale);
	  align = alignof(align, data, columns);

	  let array = [];
	  let index = [];
	  let iterator = data[Symbol.iterator]();
	  let iterindex = 0;
	  let N = lengthof(data); // total number of rows (if known)
	  let n = minlengthof(rows * 2); // number of currently-shown rows

	  // Defer materializing index and data arrays until needed.
	  function materialize() {
	    if (iterindex >= 0) {
	      iterindex = iterator = undefined;
	      index = Uint32Array.from(array = arrayify(data), (_, i) => i);
	      N = index.length;
	    }
	  }

	  function minlengthof(length) {
	    length = Math.floor(length);
	    if (N !== undefined) return Math.min(N, length);
	    if (length <= iterindex) return length;
	    while (length > iterindex) {
	      const {done, value} = iterator.next();
	      if (done) return N = iterindex;
	      index.push(iterindex++);
	      array.push(value);
	    }
	    return iterindex;
	  }

	  let currentSortHeader = null, currentReverse = false;
	  let selected = new Set();
	  let anchor = null, head = null;

	  const tbody = html`<tbody>`;
	  const tr = html`<tr><td><input type=${multiple ? "checkbox" : "radio"} name=${multiple ? null : "radio"}></td>${columns.map(() => html`<td>`)}`;
	  const theadr = html`<tr><th><input type=checkbox onclick=${reselectAll} disabled=${!multiple}></th>${columns.map((column) => html`<th title=${column} onclick=${event => resort(event, column)}><span></span>${header && column in header ? header[column] : column}</th>`)}</tr>`;
	  root.appendChild(html.fragment`<table style=${{tableLayout: layout}}>
  <thead>${minlengthof(1) || columns.length ? theadr : null}</thead>
  ${tbody}
</table>
<style>${columns.map((column, i) => {
  const rules = [];
  if (align[column] != null) rules.push(`text-align:${align[column]}`);
  if (width[column] != null) rules.push(`width:${length(width[column])}`);
  if (rules.length) return `#${id} tr>:nth-child(${i + 2}){${rules.join(";")}}`;
}).filter(identity).join("\n")}</style>`);
	  function appendRows(i, j) {
	    if (iterindex === i) {
	      for (; i < j; ++i) {
	        appendRow(iterator.next().value, i);
	      }
	      iterindex = j;
	    } else {
	      for (let k; i < j; ++i) {
	        k = index[i];
	        appendRow(array[k], k);
	      }
	    }
	  }

	  function appendRow(d, i) {
	    const itr = tr.cloneNode(true);
	    const input = inputof(itr);
	    input.onclick = reselect;
	    input.checked = selected.has(i);
	    input.value = i;
	    if (d != null) for (let j = 0; j < columns.length; ++j) {
	      let column = columns[j];
	      let value = d[column];
	      if (!defined(value)) continue;
	      value = format[column](value, i, data);
	      if (!(value instanceof Node)) value = document.createTextNode(value);
	      itr.childNodes[j + 1].appendChild(value);
	    }
	    tbody.append(itr);
	  }

	  function unselect(i) {
	    materialize();
	    let j = index.indexOf(i);
	    if (j < tbody.childNodes.length) {
	      const tr = tbody.childNodes[j];
	      inputof(tr).checked = false;
	    }
	    selected.delete(i);
	  }

	  function select(i) {
	    materialize();
	    let j = index.indexOf(i);
	    if (j < tbody.childNodes.length) {
	      const tr = tbody.childNodes[j];
	      inputof(tr).checked = true;
	    }
	    selected.add(i);
	  }

	  function* range(i, j) {
	    materialize();
	    i = index.indexOf(i), j = index.indexOf(j);
	    if (i < j) while (i <= j) yield index[i++];
	    else while (j <= i) yield index[j++];
	  }

	  function first(set) {
	    return set[Symbol.iterator]().next().value;
	  }

	  function reselectAll(event) {
	    materialize();
	    if (this.checked) {
	      selected = new Set(index);
	      for (const tr of tbody.childNodes) {
	        inputof(tr).checked = true;
	      }
	    } else {
	      for (let i of selected) unselect(i);
	      anchor = head = null;
	      if (event.detail) event.currentTarget.blur();
	    }
	    reinput();
	  }

	  function reselect(event) {
	    materialize();
	    let i = +this.value;
	    if (!multiple) {
	      for (let i of selected) unselect(i);
	      select(i);
	    } else if (event.shiftKey) {
	      if (anchor === null) anchor = selected.size ? first(selected) : index[0];
	      else for (let i of range(anchor, head)) unselect(i);
	      head = i;
	      for (let i of range(anchor, head)) select(i);
	    } else {
	      anchor = head = i;
	      if (selected.has(i)) {
	        unselect(i);
	        anchor = head = null;
	        if (event.detail) event.currentTarget.blur();
	      } else {
	        select(i);
	      }
	    }
	    reinput();
	  }

	  function resort(event, column) {
	    materialize();
	    const th = event.currentTarget;
	    let compare;
	    if (currentSortHeader === th && event.metaKey) {
	      orderof(currentSortHeader).textContent = "";
	      currentSortHeader = null;
	      currentReverse = false;
	      compare = ascending;
	    } else {
	      if (currentSortHeader === th) {
	        currentReverse = !currentReverse;
	      } else {
	        if (currentSortHeader) {
	          orderof(currentSortHeader).textContent = "";
	        }
	        currentSortHeader = th;
	        currentReverse = event.altKey;
	      }
	      const order = currentReverse ? descending : ascending;
	      compare = (a, b) => order(array[a][column], array[b][column]);
	      orderof(th).textContent = currentReverse ? "▾"  : "▴";
	    }
	    index.sort(compare);
	    selected = new Set(Array.from(selected).sort(compare));
	    root.scrollTo(root.scrollLeft, 0);
	    while (tbody.firstChild) tbody.firstChild.remove();
	    appendRows(0, n = minlengthof(rows * 2));
	    anchor = head = null;
	    reinput();
	  }

	  function reinput() {
	    const check = inputof(theadr);
	    check.disabled = !multiple && !selected.size;
	    check.indeterminate = multiple && selected.size && selected.size !== N; // assume materalized!
	    check.checked = selected.size;
	    value = undefined; // lazily computed
	  }

	  root.onscroll = () => {
	    if (root.scrollHeight - root.scrollTop < rows * rowHeight * 1.5 && n < minlengthof(n + 1)) {
	      appendRows(n, n = minlengthof(n + rows));
	    }
	  };

	  if (sort === undefined && reverse) {
	    materialize();
	    index.reverse();
	  }

	  if (value !== undefined) {
	    materialize();
	    if (multiple) {
	      const values = new Set(value);
	      selected = new Set(index.filter(i => values.has(array[i])));
	    } else {
	      const i = array.indexOf(value);
	      selected = i < 0 ? new Set() : new Set([i]);
	    }
	    reinput();
	  }

	  if (minlengthof(1)) {
	    appendRows(0, n);
	  } else {
	    tbody.append(html`<tr>${columns.length ? html`<td>` : null}<td rowspan=${columns.length} style="padding-left: var(--length3); font-style: italic;">No results.</td></tr>`);
	  }

	  if (sort !== undefined) {
	    let i = columns.indexOf(sort);
	    if (i >= 0) {
	      if (reverse) currentSortHeader = theadr.childNodes[i + 1];
	      resort({currentTarget: theadr.childNodes[i + 1]}, columns[i]);
	    }
	  }

	  return Object.defineProperty(root, "value", {
	    get() {
	      if (value === undefined) {
	        materialize();
	        if (multiple) {
	          value = Array.from(required && selected.size === 0 ? index : selected, i => array[i]);
	          value.columns = columns;
	        } else if (selected.size) {
	          const [i] = selected;
	          value = array[i];
	        } else {
	          value = null;
	        }
	      }
	      return value;
	    },
	    set(v) {
	      materialize();
	      if (multiple) {
	        const values = new Set(v);
	        const selection = new Set(index.filter(i => values.has(array[i])));
	        for (const i of selected) if (!selection.has(i)) unselect(i);
	        for (const i of selection) if (!selected.has(i)) select(i);
	      } else {
	        const i = array.indexOf(v);
	        selected = i < 0 ? new Set() : new Set([i]);
	      }
	      value = undefined; // lazily computed
	    }
	  });
	}

	function inputof(tr) {
	  return tr.firstChild.firstChild;
	}

	function orderof(th) {
	  return th.firstChild;
	}

	function formatof(base = {}, data, columns, locale) {
	  const format = Object.create(null);
	  for (const column of columns) {
	    if (column in base) {
	      format[column] = base[column];
	      continue;
	    }
	    switch (type(data, column)) {
	      case "number": format[column] = formatLocaleNumber(locale); break;
	      case "date": format[column] = formatDate; break;
	      default: format[column] = formatLocaleAuto(locale); break;
	    }
	  }
	  return format;
	}

	function alignof(base = {}, data, columns) {
	  const align = Object.create(null);
	  for (const column of columns) {
	    if (column in base) {
	      align[column] = base[column];
	    } else if (type(data, column) === "number") {
	      align[column] = "right";
	    }
	  }
	  return align;
	}

	function type(data, column) {
	  for (const d of data) {
	    if (d == null) continue;
	    const value = d[column];
	    if (value == null) continue;
	    if (typeof value === "number") return "number";
	    if (value instanceof Date) return "date";
	    return;
	  }
	}

	function lengthof(data) {
	  if (typeof data.length === "number") return data.length; // array or array-like
	  if (typeof data.size === "number") return data.size; // map, set
	  if (typeof data.numRows === "function") return data.numRows(); // arquero
	}

	function columnsof(data) {
	  if (Array.isArray(data.columns)) return data.columns; // d3-dsv, FileAttachment
	  if (data.schema && Array.isArray(data.schema.fields)) return data.schema.fields.map(f => f.name); // apache-arrow
	  if (typeof data.columnNames === "function") return data.columnNames(); // arquero
	  const columns = new Set();
	  for (const row of data) {
	    for (const name in row) {
	      columns.add(name);
	    }
	  }
	  return Array.from(columns);
	}

	// https://observablehq.com/@robinl/to-embed-in-splink-outputs@1734
	function define(runtime, observer) {
	  const main = runtime.module();
	  main.variable(observer()).define(["md"], function(md){return(
	md`# To embed in splink outputs`
	)});
	  main.variable(observer("viewof selected_cluster_id")).define("viewof selected_cluster_id", ["splink_vis_utils","cluster_unique_ids"], function(splink_vis_utils,cluster_unique_ids){return(
	splink_vis_utils.select(cluster_unique_ids, {
	  label: "Choose cluster: "
	})
	)});
	  main.variable(observer("selected_cluster_id")).define("selected_cluster_id", ["Generators", "viewof selected_cluster_id"], (G, _) => G.input(_));
	  main.variable(observer("viewof edge_colour_metric")).define("viewof edge_colour_metric", ["splink_vis_utils","raw_edges_data"], function(splink_vis_utils,raw_edges_data)
	{
	  const node_size_options = splink_vis_utils.detect_edge_colour_metrics(
	    raw_edges_data
	  );

	  let v = splink_vis_utils.select(node_size_options, {
	    label: "Choose metric for edge colour: "
	  });
	  if (node_size_options.length == 1) {
	    v.style.visibility = "hidden";
	  }
	  return v;
	}
	);
	  main.variable(observer("edge_colour_metric")).define("edge_colour_metric", ["Generators", "viewof edge_colour_metric"], (G, _) => G.input(_));
	  main.variable(observer("viewof node_size_metric")).define("viewof node_size_metric", ["splink_vis_utils","raw_nodes_data"], function(splink_vis_utils,raw_nodes_data)
	{
	  const node_size_options = splink_vis_utils.detect_node_size_metrics(
	    raw_nodes_data
	  );
	  let v = splink_vis_utils.select(node_size_options, {
	    label: "Choose metric for node size: "
	  });
	  if (node_size_options.length == 1) {
	    v.style.visibility = "hidden";
	  }
	  return v;
	}
	);
	  main.variable(observer("node_size_metric")).define("node_size_metric", ["Generators", "viewof node_size_metric"], (G, _) => G.input(_));
	  main.variable(observer("viewof node_colour_metric")).define("viewof node_colour_metric", ["splink_vis_utils","raw_nodes_data"], function(splink_vis_utils,raw_nodes_data)
	{
	  const node_size_options = splink_vis_utils.detect_node_colour_metrics(
	    raw_nodes_data
	  );

	  let v = splink_vis_utils.select(node_size_options, {
	    label: "Choose metric for node colour: "
	  });
	  if (node_size_options.length == 1) {
	    v.style.visibility = "hidden";
	  }
	  return v;
	}
	);
	  main.variable(observer("node_colour_metric")).define("node_colour_metric", ["Generators", "viewof node_colour_metric"], (G, _) => G.input(_));
	  main.variable(observer("viewof show_edge_comparison_type")).define("viewof show_edge_comparison_type", ["splink_vis_utils"], function(splink_vis_utils){return(
	splink_vis_utils.checkbox(
	  new Map([
	    ["Show waterfall chart on edge click", "show_waterfall"],
	    ["Show raw edge data on edge click", "raw_edge_data"],

	    ["Show comparison columns on edge click", "cc_data"],
	    ["Show history of node clicks", "node_history"]
	  ]),
	  {
	    label: "",
	    value: ["show_waterfall", "raw_edge_data", "node_history"]
	  }
	)
	)});
	  main.variable(observer("show_edge_comparison_type")).define("show_edge_comparison_type", ["Generators", "viewof show_edge_comparison_type"], (G, _) => G.input(_));
	  main.variable(observer("viewof show_full_tables")).define("viewof show_full_tables", ["raw_clusters_data","splink_vis_utils"], function(raw_clusters_data,splink_vis_utils)
	{
	  let options;
	  
	  if (raw_clusters_data == null) {
	    options = new Map([
	      ["Show table of all edges", "edges"],
	      ["Show table of all nodes", "nodes"]
	    ]);
	  } else {
	    options = new Map([
	      ["Show cluster info", "clusters"],
	      ["Show table of all edges", "edges"],
	      ["Show table of all nodes", "nodes"],
	      ["Show table of all clusters", "all_clusters"]
	    ]);
	  }

	  return splink_vis_utils.checkbox(options, {
	    label: ""
	  });
	}
	);
	  main.variable(observer("show_full_tables")).define("show_full_tables", ["Generators", "viewof show_full_tables"], (G, _) => G.input(_));
	  main.variable(observer("viewof score_threshold_filter")).define("viewof score_threshold_filter", ["splink_vis_utils"], function(splink_vis_utils){return(
	splink_vis_utils.range([-20, 20], {
	  label: 'Filter out edges with match weight below threshold:',
	  value: -20,
	  step: 0.1
	})
	)});
	  main.variable(observer("score_threshold_filter")).define("score_threshold_filter", ["Generators", "viewof score_threshold_filter"], (G, _) => G.input(_));
	  main.variable(observer("corresponding_probability")).define("corresponding_probability", ["html","splink_vis_utils","score_threshold_filter"], function(html,splink_vis_utils,score_threshold_filter){return(
	html`Your chosen threshold corresponds to a match probability of ${splink_vis_utils
  .log2_bayes_factor_to_prob(score_threshold_filter)
  .toPrecision(4)}`
	)});
	  main.variable(observer("viewof additional_graph_controls")).define("viewof additional_graph_controls", ["splink_vis_utils"], function(splink_vis_utils){return(
	splink_vis_utils.checkbox(
	  new Map([["Show additional graph controls", "graph_controls"]]),
	  {
	    label: ""
	  }
	)
	)});
	  main.variable(observer("additional_graph_controls")).define("additional_graph_controls", ["Generators", "viewof additional_graph_controls"], (G, _) => G.input(_));
	  main.variable(observer("edge_table")).define("edge_table", ["selected_edge","html","show_edge_comparison_type","splink_vis_utils","ss"], function(selected_edge,html,show_edge_comparison_type,splink_vis_utils,ss)
	{
	  if (selected_edge == undefined) {
	    return html``;
	  }

	  if (show_edge_comparison_type.includes("raw_edge_data")) {
	    return html`
  <h3>Rows compared by selected edge</h3>   
    ${splink_vis_utils.edge_row_to_table(selected_edge, ss)}

`;
	  }

	  return html``;
	}
	);
	  main.variable(observer("nothing_selected_message")).define("nothing_selected_message", ["no_edge_selected","no_node_selected","html","selected_edge"], function(no_edge_selected,no_node_selected,html,selected_edge)
	{
	  if (no_edge_selected && no_node_selected) {
	    return html`<span style="color:red"'>Click on nodes and/or edges in the below graph to show data and chart</span>`;
	  }

	  if (typeof selected_edge == 'undefined') {
	    return html`<span style="color:red"'>Click on an edges in the below graph to show waterfall chart and table</span>`;
	  }

	  return html``;
	}
	);
	  main.variable(observer("viewof force_directed_chart")).define("viewof force_directed_chart", ["vegaEmbed","splink_vis_utils","spec"], function(vegaEmbed,splink_vis_utils,spec){return(
	vegaEmbed(splink_vis_utils.cloneDeep(spec))
	)});
	  main.variable(observer("force_directed_chart")).define("force_directed_chart", ["Generators", "viewof force_directed_chart"], (G, _) => G.input(_));
	  main.variable(observer("node_history_table")).define("node_history_table", ["node_history","html","show_edge_comparison_type","splink_vis_utils","ss"], function(node_history,html,show_edge_comparison_type,splink_vis_utils,ss)
	{
	  if (node_history.length == 0) {
	    return html``;
	  }
	  if (show_edge_comparison_type.includes("node_history")) {
	    return html`
<h3>History of clicked nodes</h3> 
${splink_vis_utils.node_rows_to_table(node_history, ss)}
`;
	  }
	  return html``;
	}
	);
	  main.variable(observer("viewof refresh")).define("viewof refresh", ["Inputs"], function(Inputs){return(
	Inputs.button("refresh splink_vis_utils javascript lib")
	)});
	  main.variable(observer("refresh")).define("refresh", ["Generators", "viewof refresh"], (G, _) => G.input(_));
	  main.variable(observer("cluster_table")).define("cluster_table", ["selected_cluster_metrics","html","show_full_tables","splink_vis_utils"], function(selected_cluster_metrics,html,show_full_tables,splink_vis_utils)
	{
	  if (selected_cluster_metrics == null) {
	    return html``;
	  }
	  if (!show_full_tables.includes("clusters")) {
	    return html``;
	  }
	  return html`
  <h3> Cluster metrics </h3>
  ${splink_vis_utils.single_cluster_table(selected_cluster_metrics)}
`;
	}
	);
	  main.variable(observer("comparison_columns_table")).define("comparison_columns_table", ["no_edge_selected","html","show_edge_comparison_type","splink_vis_utils","selected_edge","ss"], function(no_edge_selected,html,show_edge_comparison_type,splink_vis_utils,selected_edge,ss)
	{
	  if (no_edge_selected) {
	    return html``;
	  }
	  if (show_edge_comparison_type.includes("cc_data")) {
	    return splink_vis_utils.comparison_column_table(selected_edge, ss);
	  }
	  return html``;
	}
	);
	  main.variable(observer("waterfall_chart")).define("waterfall_chart", ["no_edge_selected","html","show_edge_comparison_type","splink_vis_utils","selected_edge","ss","vegaEmbed"], function(no_edge_selected,html,show_edge_comparison_type,splink_vis_utils,selected_edge,ss,vegaEmbed)
	{
	  if (no_edge_selected) {
	    return html``;
	  } else if (!show_edge_comparison_type.includes("show_waterfall")) {
	    return html``;
	  } else {
	    let waterfall_data = splink_vis_utils.get_waterfall_chart_data(
	      selected_edge,
	      ss
	    );

	    return vegaEmbed(
	      splink_vis_utils.get_waterfall_chart_spec(waterfall_data, {})
	    );
	  }
	}
	);
	  main.variable(observer("edges_full_table")).define("edges_full_table", ["show_full_tables","html","splink_vis_utils","filtered_edges"], function(show_full_tables,html,splink_vis_utils,filtered_edges)
	{
	  if (show_full_tables.includes("edges")) {
	    // const filtered_edges = filtered_edges.filter(
	    //   d => d[svu_options.cluster_colname + "_l"] == selected_cluster_id
	    // );
	    return html`
    <h3>Edges corresponding to selected cluster, filtered by threshold</h3>
    Click column headers to sort
    
    ${splink_vis_utils.table(filtered_edges, { layout: "auto" })}

  `;
	  } else {
	    return html``;
	  }
	}
	);
	  main.variable(observer("nodes_full_table")).define("nodes_full_table", ["show_full_tables","raw_nodes_data","svu_options","selected_cluster_id","html","splink_vis_utils"], function(show_full_tables,raw_nodes_data,svu_options,selected_cluster_id,html,splink_vis_utils)
	{
	  if (show_full_tables.includes("nodes")) {
	    const filtered_nodes = raw_nodes_data.filter(
	      d => d[svu_options.cluster_colname] == selected_cluster_id
	    );

	    return html`
    <h3>All nodes corresponding to selected cluster</h3>
    Click column headers to sort
    
    ${splink_vis_utils.table(filtered_nodes, { layout: "auto" })}

  `;
	  } else {
	    return html``;
	  }
	}
	);
	  main.variable(observer("clusters_full_table")).define("clusters_full_table", ["show_full_tables","html","splink_vis_utils","raw_clusters_data"], function(show_full_tables,html,splink_vis_utils,raw_clusters_data)
	{
	  if (show_full_tables.includes("all_clusters")) {
	    return html`
    <h3>All clusters</h3>
    Click column headers to sort
    
    ${splink_vis_utils.table(raw_clusters_data, { layout: "auto" })}

  `;
	  } else {
	    return html``;
	  }
	}
	);
	  main.variable(observer()).define(["md"], function(md){return(
	md`## Outputs`
	)});
	  main.variable(observer("spec")).define("spec", ["splink_vis_utils","filtered_nodes","ss","filtered_edges","edge_colour_metric","node_size_metric","node_colour_metric","width","additional_graph_controls"], function(splink_vis_utils,filtered_nodes,ss,filtered_edges,edge_colour_metric,node_size_metric,node_colour_metric,width,additional_graph_controls)
	{
	  let formatted_nodes = splink_vis_utils.format_nodes_data_for_force_directed(
	    filtered_nodes,
	    ss
	  );

	  let formatted_edges = splink_vis_utils.format_edges_data_for_force_directed(
	    filtered_edges,
	    ss
	  );
	  let s = new splink_vis_utils.ForceDirectedChart(
	    formatted_nodes,
	    formatted_edges
	  );

	  let edge_colour_args =
	    splink_vis_utils.metric_vis_args["edge_colour"][edge_colour_metric];

	  s.set_edge_colour_metric(...Object.values(edge_colour_args));

	  if (node_size_metric != "none") {
	    let node_size_args =
	      splink_vis_utils.metric_vis_args["node_size"][node_size_metric];
	    s.set_node_area_metric(...Object.values(node_size_args));
	  }

	  if (node_colour_metric != "none") {
	    let node_size_args =
	      splink_vis_utils.metric_vis_args["node_colour"][node_colour_metric];
	    s.set_node_colour_metric(...Object.values(node_size_args));
	  }

	  s.set_height_from_nodes_data();

	  let new_width = width;
	  if (width > 1500) {
	    new_width = 1500;
	  }
	  s.set_starting_width(new_width);
	  if (additional_graph_controls.length == 0) {
	    s.remove_all_sliders();
	  }

	  return s.spec;
	}
	);
	  main.variable(observer("ss")).define("ss", ["splink_vis_utils","splink_settings"], function(splink_vis_utils,splink_settings){return(
	new splink_vis_utils.SplinkSettings(JSON.stringify(splink_settings))
	)});
	  main.variable(observer("no_edge_selected")).define("no_edge_selected", ["selected_edge"], function(selected_edge){return(
	typeof selected_edge == 'undefined'
	)});
	  main.variable(observer("no_node_selected")).define("no_node_selected", ["selected_node"], function(selected_node){return(
	typeof selected_node == 'undefined'
	)});
	  main.variable(observer()).define(["md"], function(md){return(
	md`## Data processing`
	)});
	  main.variable(observer("cluster_unique_ids")).define("cluster_unique_ids", ["splink_vis_utils","raw_nodes_data","svu_options","named_clusters"], function(splink_vis_utils,raw_nodes_data,svu_options,named_clusters)
	{
	  let cluster_ids = splink_vis_utils.get_unique_cluster_ids_from_nodes_data(
	    raw_nodes_data,
	    svu_options.cluster_colname
	  );
	  cluster_ids = cluster_ids.map(d => d.toString());

	  if (named_clusters != null) {
	    let cid_map = new Map();

	    Object.entries(named_clusters).forEach(e => {
	      cid_map.set(e[1], e[0]);

	      const index = cluster_ids.indexOf(e[0]);

	      if (index > -1) {
	        cluster_ids.splice(index, 1);
	      }
	    });

	    cluster_ids.forEach(d => cid_map.set(d, d));

	    return cid_map;
	  }
	  return cluster_ids;
	}
	);
	  main.variable(observer("selected_edge")).define("selected_edge", ["observe_chart_data","force_directed_chart"], function(observe_chart_data,force_directed_chart){return(
	observe_chart_data(force_directed_chart, "edge_click")
	)});
	  main.variable(observer("observe_chart_data")).define("observe_chart_data", ["Generators"], function(Generators){return(
	function observe_chart_data(chart, signal_name) {
	  return Generators.observe(function(notify) {
	    // change is a function; calling change triggers the resolution of the current promise with the passed value.

	    // Yield the element’s initial value.
	    const signaled = (name, value) => notify(chart.signal(signal_name));
	    chart.addSignalListener(signal_name, signaled);
	    notify(chart.signal(signal_name));

	    return () => chart.removeSignalListener(signal_name, signaled);
	  });
	}
	)});
	  main.variable(observer("selected_node")).define("selected_node", ["observe_chart_data","force_directed_chart"], function(observe_chart_data,force_directed_chart){return(
	observe_chart_data(force_directed_chart, "node_click")
	)});
	  main.variable(observer("selected_cluster_metrics")).define("selected_cluster_metrics", ["raw_clusters_data","svu_options","selected_cluster_id"], function(raw_clusters_data,svu_options,selected_cluster_id)
	{
	  if (raw_clusters_data == null) {
	    return null;
	  } else {
	    let data = raw_clusters_data.filter(
	      d => d[svu_options.cluster_colname] == selected_cluster_id
	    );
	    return data[0];
	  }
	}
	);
	  main.variable(observer("filtered_nodes")).define("filtered_nodes", ["splink_vis_utils","raw_nodes_data","svu_options","selected_cluster_id"], function(splink_vis_utils,raw_nodes_data,svu_options,selected_cluster_id){return(
	splink_vis_utils.filter_nodes_with_cluster_id(
	  raw_nodes_data,
	  svu_options.cluster_colname,
	  selected_cluster_id
	)
	)});
	  main.variable(observer("filtered_edges")).define("filtered_edges", ["splink_vis_utils","raw_edges_data","svu_options","selected_cluster_id","score_threshold_filter"], function(splink_vis_utils,raw_edges_data,svu_options,selected_cluster_id,score_threshold_filter)
	{
	  let edges = splink_vis_utils.filter_edges_with_cluster_id(
	    raw_edges_data,
	    svu_options.cluster_colname,
	    selected_cluster_id
	  );

	  edges = edges.filter(
	    d =>
	      d[svu_options.prob_colname] >=
	      splink_vis_utils.log2_bayes_factor_to_prob(score_threshold_filter)
	  );

	  return edges;
	}
	);
	  main.define("initial node_history", function(){return(
	[]
	)});
	  main.variable(observer("mutable node_history")).define("mutable node_history", ["Mutable", "initial node_history"], (M, _) => new M(_));
	  main.variable(observer("node_history")).define("node_history", ["mutable node_history"], _ => _.generator);
	  main.variable(observer("control_node_history")).define("control_node_history", ["selected_node","force_directed_chart","mutable node_history"], function(selected_node,force_directed_chart,$0)
	{

	  if (typeof force_directed_chart._signals.node_click.value == 'undefined') {
	    $0.value = [];
	  } else {
	    $0.value.unshift(
	      force_directed_chart._signals.node_click.value
	    );
	    $0.value = $0.value;
	  }
	}
	);
	  main.variable(observer()).define(["md"], function(md){return(
	md`## Following are global variables embedded in final html so not needed in final version`
	)});
	  return main;
	}

	const log2 = Math.log2;

	function bayes_factor_to_prob(b) {
	  return b / (b + 1);
	}

	function prob_to_bayes_factor(p) {
	  return p / (1 - p);
	}

	function prob_to_log2_bayes_factor(p) {
	  return log2(prob_to_bayes_factor(p));
	}

	function log2_bayes_factor_to_prob(log2_b) {
	  return bayes_factor_to_prob(2 ** log2_b);
	}

	function get_waterfall_row_single_column(
	  gamma_key,
	  row,
	  splink_settings,
	  term_freqs
	) {
	  let key = gamma_key;
	  let gamma_value = row[key];
	  let col_name = key.replace("gamma_", "");

	  let this_cc = splink_settings.get_col_by_name(col_name);

	  let value_l = row[col_name + "_l"];
	  let value_r = row[col_name + "_r"];

	  let u_probability, m_probability;
	  if (gamma_value == -1) {
	    u_probability = 0.5;
	    m_probability = 0.5;
	  } else {
	    u_probability = this_cc.u_probabilities[gamma_value];
	    m_probability = this_cc.m_probabilities[gamma_value];

	    if (value_l == value_r) {
	      if (col_name in term_freqs) {
	        let tfs = term_freqs[col_name];
	        u_probability = tfs[value_l] || u_probability;
	      }
	    }
	  }

	  let bayes_factor = m_probability / u_probability;

	  return {
	    bayes_factor: bayes_factor,
	    column_name: col_name,
	    gamma_column_name: "𝛾_" + col_name,
	    gamma_index: gamma_value,

	    level_name: "level_" + gamma_value,

	    log2_bayes_factor: log2(bayes_factor),
	    m_probability: m_probability,

	    num_levels: null,
	    u_probability: u_probability,
	    value_l: value_l,
	    value_r: value_r,
	  };
	}

	function get_waterfall_data_comparison_columns(
	  row,
	  splink_settings,
	  term_freqs
	) {
	  let keys = Object.keys(row);
	  keys = keys.filter((key) => key.includes("gamma_"));
	  return keys.map((gamma_key) =>
	    get_waterfall_row_single_column(gamma_key, row, splink_settings, term_freqs)
	  );
	}

	function get_waterfall_data_lambda_row(splink_settings) {
	  let row = {
	    bayes_factor: prob_to_bayes_factor(
	      splink_settings.settings_dict.proportion_of_matches
	    ),
	    column_name: "Prior",
	    gamma_column_name: "",
	    gamma_index: "",

	    level_name: null,

	    log2_bayes_factor: prob_to_log2_bayes_factor(
	      splink_settings.settings_dict.proportion_of_matches
	    ),
	    m_probability: null,

	    num_levels: null,
	    u_probability: null,
	    value_l: "",
	    value_r: "",
	  };

	  return row;
	}

	function get_waterfall_data_final_row() {
	  let row = {
	    bayes_factor: null,
	    column_name: "Final score",
	    gamma_column_name: "",
	    gamma_index: "",

	    level_name: null,

	    log2_bayes_factor: null,
	    m_probability: null,

	    num_levels: null,
	    u_probability: null,
	    value_l: "",
	    value_r: "",
	  };

	  return row;
	}

	function get_waterfall_chart_data(
	  row,
	  splink_settings,
	  term_freqs = {}
	) {
	  let lambda_row = get_waterfall_data_lambda_row(splink_settings);
	  let waterfall_data_other_rows = get_waterfall_data_comparison_columns(
	    row,
	    splink_settings,
	    term_freqs
	  );

	  let rows_except_final = [lambda_row].concat(waterfall_data_other_rows);
	  let final_row = get_waterfall_data_final_row();

	  let cumulative_log2_bayes_factor = rows_except_final.reduce(
	    (a, b) => a + b["log2_bayes_factor"],
	    0
	  );

	  final_row["bayes_factor"] = 2 ** cumulative_log2_bayes_factor;
	  final_row["log2_bayes_factor"] = cumulative_log2_bayes_factor;

	  return rows_except_final.concat([final_row]);
	}

	var $schema$1 = "https://vega.github.io/schema/vega/v5.json";
	var description = "Links and nodes";
	var padding = 0;
	var autosize = "none";
	var signals = [
		{
			name: "node_click",
			on: [
				{
					events: "@nodes:click",
					update: "datum"
				}
			]
		},
		{
			name: "nodeRadius",
			value: 1,
			bind: {
				input: "range",
				min: 0.2,
				max: 4,
				step: 0.1
			}
		},
		{
			name: "nodeCollideStrength",
			value: 1,
			bind: {
				input: "range",
				min: 0.2,
				max: 4,
				step: 0.1
			}
		},
		{
			name: "nodeCollideRadius",
			value: 1.4,
			bind: {
				input: "range",
				min: 0.2,
				max: 4,
				step: 0.1
			}
		},
		{
			name: "linkStrength",
			value: 0.5,
			bind: {
				input: "range",
				min: 0,
				max: 2,
				step: 0.01
			}
		},
		{
			name: "edge_click",
			on: [
				{
					events: "@edges:click",
					update: "datum"
				}
			]
		},
		{
			name: "cx",
			update: "width / 2"
		},
		{
			name: "cy",
			update: "height / 2"
		},
		{
			name: "nodeCharge",
			value: 30,
			bind: {
				input: "range",
				min: -2000,
				max: 500,
				step: 1
			}
		},
		{
			name: "linkDistance",
			value: 0.5,
			bind: {
				input: "range",
				min: 0.1,
				max: 2,
				step: 0.1
			}
		},
		{
			name: "vis_height",
			value: 200,
			bind: {
				input: "range",
				min: 400,
				max: 2000,
				step: 50
			}
		},
		{
			name: "vis_width",
			value: 1000,
			bind: {
				input: "range",
				min: 400,
				max: 2000,
				step: 20
			}
		},
		{
			name: "static",
			value: true,
			bind: {
				input: "checkbox"
			}
		},
		{
			description: "State variable for active node fix status.",
			name: "fix",
			value: false,
			on: [
				{
					events: "symbol:mouseout[!event.buttons], window:mouseup",
					update: "false"
				},
				{
					events: "symbol:mouseover",
					update: "fix || true"
				},
				{
					events: "[symbol:mousedown, window:mouseup] > window:mousemove!",
					update: "xy()",
					force: true
				}
			]
		},
		{
			description: "Graph node most recently interacted with.",
			name: "node",
			value: null,
			on: [
				{
					events: "symbol:mouseover",
					update: "fix === true ? item() : node"
				}
			]
		},
		{
			description: "Flag to restart Force simulation upon data changes.",
			name: "restart",
			value: false,
			on: [
				{
					events: {
						signal: "fix"
					},
					update: "fix && fix.length"
				}
			]
		}
	];
	var width$1 = {
		signal: "vis_width"
	};
	var height$1 = {
		signal: "vis_height"
	};
	var data$1 = [
		{
			name: "node-data",
			values: null
		},
		{
			name: "link-data",
			values: null
		}
	];
	var scales = [
		{
			name: "color",
			type: "ordinal",
			domain: {
				data: "node-data",
				field: "cluster_id"
			},
			range: {
				scheme: "category20c"
			}
		}
	];
	var legends = [
	];
	var marks = [
		{
			name: "nodes",
			type: "symbol",
			zindex: 1,
			from: {
				data: "node-data"
			},
			on: [
				{
					trigger: "fix",
					modify: "node",
					values: "fix === true ? {fx: node.x, fy: node.y} : {fx: fix[0], fy: fix[1]}"
				}
			],
			encode: {
				enter: {
					stroke: {
						value: "black"
					},
					tooltip: {
						signal: "datum.tooltip"
					}
				},
				update: {
					size: {
						value: 1000,
						mult: {
							signal: "nodeRadius"
						}
					},
					cursor: {
						value: "pointer"
					},
					fill: {
						scale: "color",
						field: "cluster_id"
					}
				}
			},
			transform: [
				{
					type: "force",
					iterations: 400,
					restart: {
						signal: "restart"
					},
					"static": {
						signal: "static"
					},
					signal: "force",
					forces: [
						{
							force: "center",
							x: {
								signal: "cx"
							},
							y: {
								signal: "cy"
							}
						},
						{
							force: "collide",
							radius: {
								expr: "pow(1000*nodeRadius,0.5)*nodeCollideStrength*nodeCollideRadius"
							},
							strength: {
								signal: "nodeCollideStrength"
							}
						},
						{
							force: "nbody",
							strength: {
								signal: "nodeCharge"
							}
						},
						{
							description: "Uses link-data to find links between nodes constraining x and y of nodes.  Tranforms link-data so source and target are objects that include e.g. x and y coords",
							force: "link",
							links: "link-data",
							distance: {
								expr: "50*linkDistance"
							},
							id: "datum.__node_id",
							strength: {
								signal: "linkStrength"
							}
						}
					]
				}
			]
		},
		{
			description: "The force link transform will replace source and target with objects containing x and y coords.  We need to extract x and y to plot a path between them",
			type: "path",
			name: "edges",
			from: {
				data: "link-data"
			},
			interactive: true,
			encode: {
				update: {
					stroke: {
						value: "black"
					},
					tooltip: {
						signal: "datum.tooltip"
					},
					strokeWidth: {
						value: 2
					}
				}
			},
			transform: [
				{
					type: "linkpath",
					require: {
						signal: "force"
					},
					shape: "line",
					sourceX: "datum.source.x",
					sourceY: "datum.source.y",
					targetX: "datum.target.x",
					targetY: "datum.target.y"
				}
			]
		},
		{
			type: "text",
			from: {
				data: "nodes"
			},
			interactive: false,
			zindex: 2,
			encode: {
				enter: {
					align: {
						value: "center"
					},
					baseline: {
						value: "middle"
					},
					fontSize: {
						value: 12
					},
					fontWeight: {
						value: "bold"
					},
					text: {
						field: "datum.__node_id"
					}
				},
				update: {
					x: {
						field: "x"
					},
					y: {
						field: "y"
					}
				}
			}
		}
	];
	var base_spec$1 = {
		$schema: $schema$1,
		description: description,
		padding: padding,
		autosize: autosize,
		signals: signals,
		width: width$1,
		height: height$1,
		data: data$1,
		scales: scales,
		legends: legends,
		marks: marks
	};

	function find_obj_in_list(list, key, value) {
	  return list.find(function (item) {
	    if (item[key] === value) {
	      return true;
	    }
	  });
	}

	function replace_in_list_or_push(list, key, value, obj) {
	  const foundIndex = list.findIndex(function (item) {
	    if (item[key] === value) {
	      return true;
	    }
	  });

	  if (foundIndex == -1) {
	    list.push(obj);
	  } else {
	    list[foundIndex] = obj;
	  }
	}

	class ForceDirectedChart {
	  constructor(nodes_data, links_data) {
	    let base_spec_cp = cloneDeep(base_spec$1);
	    this.spec = base_spec_cp;
	    this.set_force_directed_node_data(nodes_data);
	    this.set_force_directed_edge_data(links_data);
	    this.nodes_data = nodes_data;
	  }

	  set_force_directed_node_data(data) {
	    let obj = find_obj_in_list(this.spec.data, "name", "node-data");
	    obj["values"] = data;
	  }

	  set_force_directed_edge_data(data) {
	    let obj = find_obj_in_list(this.spec.data, "name", "link-data");
	    obj["values"] = data;
	  }

	  set_edge_colour_metric(
	    edge_metric_name,
	    reverse = false,
	    domain = null,
	    range = null
	  ) {
	    if (domain == null) {
	      domain = { data: "link-data", field: edge_metric_name };
	    }

	    if (range == null) {
	      range = { scheme: "redyellowgreen" };
	    }
	    const new_link_scale = {
	      name: "link_colour",
	      type: "linear",
	      domain: domain,
	      range: range,
	      reverse: reverse,
	    };

	    replace_in_list_or_push(
	      this.spec.scales,
	      "name",
	      "link_colour",
	      new_link_scale
	    );

	    let link_mark = find_obj_in_list(this.spec.marks, "name", "edges");

	    link_mark.encode.update.stroke = {
	      scale: "link_colour",
	      field: edge_metric_name,
	    };
	  }

	  set_edge_thickness_metric(edge_metric_name, reverse = false) {
	    const new_thickness_scale = {
	      name: "link_thickness",
	      type: "linear",
	      domain: { data: "link-data", field: edge_metric_name },
	      range: [0.5, 5],
	      reverse: reverse,
	    };

	    replace_in_list_or_push(
	      this.spec.scales,
	      "name",
	      "link_thickness",
	      new_thickness_scale
	    );

	    let link_mark = find_obj_in_list(this.spec.marks, "name", "edges");

	    link_mark.encode.update.strokeWidth = {
	      scale: "link_thickness",
	      field: edge_metric_name,
	    };
	  }

	  set_edge_length_metric(edge_metric_name, reverse = false) {
	    const new_edge_length_scale = {
	      name: "edge_length_scale",
	      type: "linear",
	      domain: { data: "link-data", field: edge_metric_name },
	      range: [50, 200],
	      reverse: reverse,
	    };

	    replace_in_list_or_push(
	      this.spec.scales,
	      "name",
	      "edge_length_scale",
	      new_edge_length_scale
	    );

	    const new_force = {
	      force: "link",
	      id: "datum.__node_id",
	      links: "link-data",
	      distance: {
	        expr: `scale('edge_length_scale',datum.${edge_metric_name})*linkDistance`,
	      },
	    };

	    let link_mark = find_obj_in_list(this.spec.marks, "name", "nodes");
	    let force_transform = find_obj_in_list(
	      link_mark.transform,
	      "type",
	      "force"
	    );
	    replace_in_list_or_push(force_transform.forces, "force", "link", new_force);
	  }

	  set_node_area_metric(
	    node_metric_name,
	    reverse = false,
	    domain = null,
	    range = null
	  ) {
	    if (domain == null) {
	      domain = { data: "node-data", field: node_metric_name };
	    }

	    if (range == null) {
	      range = [400, 2000];
	    }

	    const new_node_area_scale = {
	      name: "node_area_scale",
	      type: "linear",
	      nice: false,
	      reverse: reverse,
	      domain: domain,
	      range: range,
	    };

	    replace_in_list_or_push(
	      this.spec.scales,
	      "name",
	      "node_area_scale",
	      new_node_area_scale
	    );

	    let node_mark = find_obj_in_list(this.spec.marks, "name", "nodes");

	    node_mark.encode.update.size = {
	      scale: "node_area_scale",
	      field: node_metric_name,
	      mult: { signal: "nodeRadius" },
	    };

	    let force_transform = find_obj_in_list(
	      node_mark.transform,
	      "type",
	      "force"
	    );
	    let force_collide = find_obj_in_list(
	      force_transform.forces,
	      "force",
	      "collide"
	    );
	    force_collide.radius.expr = `pow(scale('node_area_scale',datum.datum.${node_metric_name})*nodeRadius,0.5)`;
	  }

	  set_node_colour_metric(
	    node_metric_name,

	    domain = null,
	    range = null
	  ) {
	    if (domain == null) {
	      domain = { data: "node-data", field: node_metric_name };
	    }

	    if (range == null) {
	      range = {
	        scheme: "category10",
	      };
	    }

	    const new_node_colour_scale = {
	      name: "node_colour_scale",
	      type: "ordinal",

	      domain: domain,
	      range: range,
	    };

	    replace_in_list_or_push(
	      this.spec.scales,
	      "name",
	      "node_colour_scale",
	      new_node_colour_scale
	    );

	    let node_mark = find_obj_in_list(this.spec.marks, "name", "nodes");

	    node_mark.encode.update.fill = {
	      scale: "node_colour_scale",
	      field: node_metric_name,
	    };
	  }

	  set_height_from_nodes_data() {
	    const min_height = 200;
	    const node_height = 150;
	    const num_nodes = this.nodes_data.length;
	    const sqrt_nodes = Math.sqrt(num_nodes);
	    let height = sqrt_nodes * node_height;
	    height = height + 20;
	    height = Math.max(min_height, height);

	    let height_signal = find_obj_in_list(
	      this.spec.signals,
	      "name",
	      "vis_height"
	    );

	    height_signal.value = height;
	  }

	  set_starting_width(new_width) {
	    let width_signal = find_obj_in_list(this.spec.signals, "name", "vis_width");
	    width_signal.value = new_width;
	  }

	  remove_all_sliders() {
	    this.spec.signals.forEach((signal) => {
	      delete signal.bind;
	    });
	  }
	}

	class ComparisonColumn {
	  constructor(cc) {
	    this.original_dict = cc;
	  }

	  get name() {
	    if (this.is_custom_column) {
	      return this.original_dict["custom_name"];
	    } else {
	      return this.original_dict["col_name"];
	    }
	  }

	  get is_custom_column() {
	    return "custom_name" in this.original_dict;
	  }

	  get columns_used() {
	    if (this.is_custom_column) {
	      return this.original_dict.custom_columns_used;
	    } else {
	      return [this.original_dict.col_name];
	    }
	  }

	  get column_case_expression_lookup() {
	    let expr = this.original_dict["case_expression"];
	    let parsed = parse_case_expression(expr);
	    let fmt = format_parsed_expression(parsed);
	    if (!("0" in fmt)) {
	      fmt["0"] = "else 0";
	    }
	    return fmt;
	  }

	  get_case_expression_for_level(level) {
	    return this.column_case_expression_lookup[level];
	  }

	  get m_probabilities() {
	    return this.original_dict["m_probabilities"];
	  }

	  get u_probabilities() {
	    return this.original_dict["u_probabilities"];
	  }

	  data_from_row(edge_row_as_dict) {
	    let data = {
	      left: [],
	      right: [],
	    };
	    this.columns_used.forEach((col) => {
	      let left_data = {
	        col_name: col,
	        col_value: edge_row_as_dict[`${col}_l`],
	      };
	      let right_data = {
	        col_name: col,
	        col_value: edge_row_as_dict[`${col}_r`],
	      };
	      data["left"].push(left_data);
	      data["right"].push(right_data);
	    });
	    return data;
	  }

	  concat_data_from_row(edge_row_as_dict) {
	    let left_right_data = this.data_from_row(edge_row_as_dict);

	    let left_data = left_right_data["left"];
	    let right_data = left_right_data["right"];

	    left_data = left_data.map((d) => d.col_value);
	    left_data = left_data.filter((d) => d != null);
	    left_data = left_data.join(" | ");

	    right_data = right_data.map((d) => d.col_value);
	    right_data = right_data.filter((d) => d != null);
	    right_data = right_data.join(" | ");

	    return {
	      left: left_data,
	      right: right_data,
	    };
	  }

	  level_from_row(edge_row_as_dict) {
	    let key = "gamma_" + this.name;
	    return edge_row_as_dict[key];
	  }

	  case_expression_from_row(edge_row_as_dict) {
	    let lev = this.level_from_row(edge_row_as_dict);
	    return this.get_case_expression_for_level(lev);
	  }
	}

	class SplinkSettings {
	  constructor(settings_json) {
	    const s = JSON.parse(settings_json);
	    this.settings_dict = s;
	  }

	  get comparison_columns() {
	    let ccs = this.settings_dict["comparison_columns"];
	    return ccs.map((d) => {
	      return new ComparisonColumn(d);
	    });
	  }

	  get comparison_column_lookup() {
	    let lookup = {};

	    this.comparison_columns.forEach((cc) => {
	      lookup[cc.name] = cc;
	    });

	    return lookup;
	  }

	  get cols_used_by_model() {
	    const ccs = this.comparison_columns;
	    let cols_in_use = [];
	    ccs.forEach((cc) => {
	      cc.columns_used.forEach((used_col) => {
	        if (cols_in_use.indexOf(used_col) == -1) {
	          cols_in_use.push(used_col);
	        }
	      });
	    });
	    return cols_in_use;
	  }

	  get cols_used_by_model_inc_add_to_retain() {
	    let all_cols = [];
	    if (this.settings_dict.link_type == "dedupe_only") {
	      all_cols.push(this.settings_dict.unique_id_column_name);
	    } else {
	      all_cols.push(this.settings_dict.unique_id_column_name);
	      all_cols.push(this.settings_dict.source_dataset_column_name);
	    }

	    let ccs = this.cols_used_by_model;
	    all_cols.push(...ccs);

	    if ("additional_columns_to_retain" in this.settings_dict) {
	      all_cols.push(...this.settings_dict["additional_columns_to_retain"]);
	    }

	    let cols_in_order_deduped = [];
	    all_cols.forEach((col) => {
	      if (cols_in_order_deduped.indexOf(col) == -1) {
	        cols_in_order_deduped.push(col);
	      }
	    });

	    return cols_in_order_deduped;
	  }

	  get_col_by_name(col_name) {
	    return this.comparison_column_lookup[col_name];
	  }
	}

	function parse_case_expression(case_expr) {
	  const case_regex = /when[\s\S]+?then[\s\S]+?(\-?[012345678])/gi;
	  let matches = case_expr.matchAll(case_regex);
	  matches = [...matches];

	  let results = {};
	  matches.forEach((d) => {
	    const key = d[1];

	    if (key in results) {
	      results[key].push(d[0]);
	    } else {
	      results[key] = [d[0]];
	    }
	  });

	  return results;
	}

	function format_parsed_expression(parsed_case_expression) {
	  let formatted_expressions = {};

	  Object.entries(parsed_case_expression).forEach((k) => {
	    formatted_expressions[k[0]] = k[1].join("\n");
	  });

	  return formatted_expressions;
	}

	function format_nodes_data_for_force_directed(
	  nodes_data,
	  splink_settings
	) {
	  // Create a __node_id field that uniquely identifies the row
	  if (splink_settings.settings_dict.link_type == "dedupe_only") {
	    let c = splink_settings.settings_dict.unique_id_column_name;
	    nodes_data.forEach(function (node) {
	      node.__node_id = node[c];
	    });
	  } else {
	    let c = splink_settings.settings_dict.unique_id_column_name;
	    let sds = splink_settings.settings_dict.source_dataset_column_name;
	    nodes_data.forEach(function (node) {
	      node.__node_id = node[sds] + "-__-" + node[c];
	    });
	  }

	  // Create a tooltip field that contains only the info used by the model
	  let cols_for_tooltip = splink_settings.cols_used_by_model_inc_add_to_retain;

	  nodes_data.forEach(function (node) {
	    let tooltip = {};
	    cols_for_tooltip.forEach(function (col) {
	      if (node[col] != null) {
	        tooltip[col] = node[col];
	      }
	    });
	    node.tooltip = tooltip;
	  });

	  return nodes_data;
	}

	function format_edges_data_for_force_directed(
	  edge_data,
	  splink_settings
	) {
	  if (splink_settings.settings_dict.link_type == "dedupe_only") {
	    let c = splink_settings.settings_dict.unique_id_column_name;
	    edge_data.forEach(function (edge) {
	      edge.source = edge[`${c}_l`];
	      edge.target = edge[`${c}_r`];
	    });
	  } else {
	    let c = splink_settings.settings_dict.unique_id_column_name;
	    let sds = splink_settings.settings_dict.source_dataset_column_name;
	    edge_data.forEach(function (edge) {
	      edge.source = edge[`${sds}_l`] + "-__-" + edge[`${c}_l`];
	      edge.target = edge[`${sds}_r`] + "-__-" + edge[`${c}_r`];
	    });
	  }

	  // Create a tooltip field that contains only the info used by the model
	  let cols_for_tooltip = splink_settings.cols_used_by_model_inc_add_to_retain;

	  let additional_cols = [
	    "match_probability",
	    "tf_adjusted_match_prob",
	    "match_weight",
	  ];

	  additional_cols = additional_cols.filter((col) => {
	    return col in edge_data[0];
	  });

	  edge_data.forEach(function (edge) {
	    let tooltip = {};
	    cols_for_tooltip.forEach(function (col) {
	      if (edge[`${col}_l`] && edge[`${col}_r`]) {
	        tooltip[`${col}_l`] = edge[`${col}_l`];
	        tooltip[`${col}_r`] = edge[`${col}_r`];
	      }
	    });
	    additional_cols.forEach((d) => (tooltip[d] = edge[d]));
	    edge.tooltip = tooltip;
	  });

	  return edge_data;
	}

	function get_unique_cluster_ids_from_nodes_data(
	  nodes_data,
	  cluster_field
	) {
	  let cluster_ids = nodes_data.map((d) => d[cluster_field]);
	  return [...new Set(cluster_ids)];
	}

	function filter_nodes_with_cluster_id(
	  nodes_data,
	  cluster_field,
	  selected_cluster_id
	) {
	  return nodes_data.filter((d) => d[cluster_field] == selected_cluster_id);
	}

	function filter_edges_with_cluster_id(
	  edges_data,
	  cluster_field,
	  selected_cluster_id
	) {
	  return edges_data
	    .filter((d) => d[`${cluster_field}_l`] == selected_cluster_id)
	    .filter((d) => d[`${cluster_field}_r`] == selected_cluster_id);
	}

	var config = {
		view: {
			continuousWidth: 400,
			continuousHeight: 300
		}
	};
	var title = {
		text: "Bayes factor intuition chart",
		subtitle: "How each comparison column contributes to the final match score"
	};
	var transform = [
		{
			filter: "(datum.bayes_factor !== 1.0)"
		},
		{
			window: [
				{
					op: "sum",
					field: "log2_bayes_factor",
					as: "sum"
				},
				{
					op: "lead",
					field: "column_name",
					as: "lead"
				}
			],
			frame: [
				null,
				0
			]
		},
		{
			calculate: "datum.column_name === \"Final score\" ? datum.sum - datum.log2_bayes_factor : datum.sum",
			as: "sum"
		},
		{
			calculate: "datum.lead === null ? datum.column_name : datum.lead",
			as: "lead"
		},
		{
			calculate: "datum.column_name === \"Final score\" || datum.column_name === \"Prior lambda\" ? 0 : datum.sum - datum.log2_bayes_factor",
			as: "previous_sum"
		},
		{
			calculate: "datum.sum > datum.previous_sum ? datum.column_name : \"\"",
			as: "top_label"
		},
		{
			calculate: "datum.sum < datum.previous_sum ? datum.column_name : \"\"",
			as: "bottom_label"
		},
		{
			calculate: "datum.sum > datum.previous_sum ? datum.sum : datum.previous_sum",
			as: "sum_top"
		},
		{
			calculate: "datum.sum < datum.previous_sum ? datum.sum : datum.previous_sum",
			as: "sum_bottom"
		},
		{
			calculate: "(datum.sum + datum.previous_sum) / 2",
			as: "center"
		},
		{
			calculate: "(datum.log2_bayes_factor > 0 ? \"+\" : \"\") + datum.log2_bayes_factor",
			as: "text_log2_bayes_factor"
		},
		{
			calculate: "datum.sum < datum.previous_sum ? 4 : -4",
			as: "dy"
		},
		{
			calculate: "datum.sum < datum.previous_sum ? \"top\" : \"bottom\"",
			as: "baseline"
		},
		{
			calculate: "1. / (1 + pow(2, -1.*datum.sum))",
			as: "prob"
		},
		{
			calculate: "0*datum.sum",
			as: "zero"
		}
	];
	var layer = [
		{
			layer: [
				{
					mark: "rule",
					encoding: {
						y: {
							field: "zero",
							type: "quantitative"
						},
						size: {
							value: 0.5
						},
						color: {
							value: "black"
						}
					}
				},
				{
					mark: {
						type: "bar",
						width: 60
					},
					encoding: {
						color: {
							condition: {
								value: "red",
								test: "(datum.log2_bayes_factor < 0)"
							},
							value: "green"
						},
						opacity: {
							condition: {
								value: 1,
								test: "datum.column_name == 'Prior lambda' || datum.column_name == 'Final score'"
							},
							value: 0.5
						},
						tooltip: [
							{
								type: "nominal",
								field: "column_name",
								title: "Comparison column"
							},
							{
								type: "nominal",
								field: "value_l",
								title: "Value (L)"
							},
							{
								type: "nominal",
								field: "value_r",
								title: "Value (R)"
							},
							{
								type: "nominal",
								field: "gamma_index",
								title: "Gamma level"
							},
							{
								type: "nominal",
								field: "max_gamma_index",
								title: "Max gamma level"
							},
							{
								type: "quantitative",
								field: "bayes_factor",
								format: ".3r",
								title: "Bayes factor"
							},
							{
								type: "quantitative",
								field: "log2_bayes_factor",
								format: ".3r",
								title: "log2(Bayes factor)"
							},
							{
								type: "quantitative",
								field: "prob",
								format: ".3r",
								title: "Adjusted match score"
							}
						],
						x: {
							type: "nominal",
							axis: {
								labelExpr: "datum.value == 'Prior lambda' || datum.value == 'Final score' ? '' : datum.value",
								labelAngle: -20,
								labelAlign: "center",
								labelPadding: 10,
								title: "Column",
								grid: true,
								tickBand: "extent"
							},
							field: "column_name",
							sort: null
						},
						y: {
							type: "quantitative",
							axis: {
								grid: false,
								orient: "left",
								title: "log2(Bayes factor)"
							},
							field: "previous_sum"
						},
						y2: {
							field: "sum"
						}
					}
				},
				{
					mark: {
						type: "text",
						fontWeight: "bold"
					},
					encoding: {
						color: {
							value: "white"
						},
						text: {
							condition: {
								type: "nominal",
								field: "log2_bayes_factor",
								format: ".2f",
								test: "abs(datum.log2_bayes_factor) > 1"
							},
							value: ""
						},
						x: {
							type: "nominal",
							axis: {
								labelAngle: 0,
								title: "Column"
							},
							field: "column_name",
							sort: null
						},
						y: {
							type: "quantitative",
							axis: {
								orient: "left"
							},
							field: "center"
						}
					}
				},
				{
					mark: {
						type: "text",
						baseline: "bottom",
						dy: -5,
						fontWeight: "bold"
					},
					encoding: {
						color: {
							value: "black"
						},
						text: {
							condition: {
								type: "nominal",
								field: "top_label",
								test: "abs(datum.log2_bayes_factor) > 1"
							},
							value: ""
						},
						x: {
							type: "nominal",
							axis: {
								labelAngle: 0,
								title: "Column"
							},
							field: "column_name",
							sort: null
						},
						y: {
							type: "quantitative",
							field: "sum_top"
						}
					}
				},
				{
					mark: {
						type: "text",
						baseline: "top",
						dy: 5,
						fontWeight: "bold"
					},
					encoding: {
						color: {
							value: "black"
						},
						text: {
							condition: {
								type: "nominal",
								field: "bottom_label",
								test: "abs(datum.log2_bayes_factor) > 1"
							},
							value: ""
						},
						x: {
							type: "nominal",
							axis: {
								labelAngle: 0,
								title: "Column"
							},
							field: "column_name",
							sort: null
						},
						y: {
							type: "quantitative",
							field: "sum_bottom"
						}
					}
				}
			]
		},
		{
			mark: {
				type: "rule",
				color: "black",
				strokeWidth: 2,
				x2Offset: 30,
				xOffset: -30
			},
			encoding: {
				x: {
					type: "nominal",
					axis: {
						labelAngle: 0,
						title: "Column"
					},
					field: "column_name",
					sort: null
				},
				x2: {
					field: "lead"
				},
				y: {
					type: "quantitative",
					axis: {
						labelExpr: "format(1 / (1 + pow(2, -1*datum.value)), '.2r')",
						orient: "right",
						title: "Probability"
					},
					field: "sum",
					scale: {
						zero: false
					}
				}
			}
		}
	];
	var height = 450;
	var resolve = {
		axis: {
			y: "independent"
		}
	};
	var width = {
		step: 75
	};
	var $schema = "https://vega.github.io/schema/vega-lite/v4.8.1.json";
	var data = {
		values: null
	};
	var waterfall = {
		config: config,
		title: title,
		transform: transform,
		layer: layer,
		height: height,
		resolve: resolve,
		width: width,
		$schema: $schema,
		data: data
	};

	var base_spec = /*#__PURE__*/Object.freeze({
		__proto__: null,
		config: config,
		title: title,
		transform: transform,
		layer: layer,
		height: height,
		resolve: resolve,
		width: width,
		$schema: $schema,
		data: data,
		'default': waterfall
	});

	function get_waterfall_chart_spec(data, overrides, simplified = false) {
	  let base_spec_2 = cloneDeep(base_spec);

	  base_spec_2.data.values = data;
	  let spec = { ...base_spec_2, ...overrides };
	  if (simplified) {
	    // Remove right hand axis
	    spec["layer"][1]["encoding"]["y"]["axis"] = false;

	    // Remove bayes factor text overlays
	    // spec["layer"][0]["layer"].splice(2, 1);
	    spec["layer"][0]["layer"][2]["encoding"]["text"] = {
	      type: "nominal",
	      field: "up_down_emoji",
	    };
	    spec["layer"][0]["layer"][2]["encoding"]["opacity"] = {
	      condition: {
	        value: 0,
	        test: "datum.column_name == 'Final score' || datum.column_name == 'Prior'",
	      },
	    };

	    // Make left hand side axis probability
	    let expr = "format(1 / (1 + pow(2, -1*datum.value)), '.2r')";
	    spec["layer"][0]["layer"][1]["encoding"]["y"]["axis"]["labelExpr"] = expr;
	    spec["layer"][0]["layer"][1]["encoding"]["y"]["axis"]["title"] =
	      "probability";

	    // Tooltip

	    spec["layer"][0]["layer"][1]["encoding"]["tooltip"] = [
	      {
	        type: "quantitative",
	        field: "prob",
	        format: ".3r",
	        title: "Cumulative match probability",
	      },
	    ];
	  }

	  return spec;
	}

	function table(...args) {
	  let tab = table$1(...args);
	  tab.removeAttribute("style");
	  return tab;
	}

	function node_row_to_table(node_as_dict, splink_settings) {
	  const first_cols = splink_settings.cols_used_by_model_inc_add_to_retain;
	  let all_cols = Object.keys(node_as_dict);

	  all_cols = all_cols.filter(function (el) {
	    return !first_cols.includes(el);
	  });

	  let cols = first_cols.concat(all_cols);

	  let d2 = {};
	  cols.forEach((c) => {
	    d2[c] = node_as_dict[c];
	  });

	  return table([d2], { layout: "auto" });
	}

	function edge_row_to_table(edge_as_dict, splink_settings) {
	  const cols_in_use = splink_settings.cols_used_by_model;
	  const row_1 = {};
	  const row_2 = {};

	  let col_priority = { 2: [], 1: [], 0: [] };

	  cols_in_use.forEach((col) => {
	    let l_val = edge_as_dict[col + "_l"];
	    let r_val = edge_as_dict[col + "_r"];

	    row_1[col] = edge_as_dict[col + "_l"];
	    row_2[col] = edge_as_dict[col + "_r"];

	    if (l_val && r_val) {
	      col_priority[2].push(col);
	    } else if (l_val || r_val) {
	      col_priority[1].push(col);
	    } else {
	      col_priority[0].push(col);
	    }
	  });

	  col_priority = col_priority[2]
	    .concat(col_priority[1])
	    .concat(col_priority[0]);

	  let row_1_ordered = {};
	  let row_2_ordered = {};
	  col_priority.forEach((col) => {
	    row_1_ordered[col] = row_1[col];
	    row_2_ordered[col] = row_2[col];
	  });
	  let table_data = [row_1_ordered, row_2_ordered];

	  return table(table_data, { layout: "auto" });
	}

	function comparison_column_table(edge_as_dict, splink_settings) {
	  // let splink_settings = new SplinkSettings
	  let ccs = splink_settings.comparison_columns;

	  let rows = [];

	  ccs.forEach((cc) => {
	    let this_row = {};
	    this_row["comparison_column_name"] = cc.name;

	    let expr = cc.case_expression_from_row(edge_as_dict);
	    let data = cc.concat_data_from_row(edge_as_dict);

	    this_row["data_left"] = data["left"];

	    this_row["data_right"] = data["right"];
	    this_row["case expression"] = expr;

	    rows.push(this_row);
	  });
	  return table(rows, { layout: "auto" });
	}

	function single_cluster_table(cluster_as_dict) {
	  let rows = [];
	  rows.push(cluster_as_dict);
	  return table(rows, { layout: "auto" });
	}

	function detect_node_size_metrics(data) {
	  const node_metrics = new Map([["None", "none"]]);
	  const keys = Object.keys(data[0]);

	  if (keys.includes("eigen_centrality")) {
	    node_metrics.set("Eigen Centrality", "eigen_centrality");
	  }

	  return node_metrics;
	}

	function detect_node_colour_metrics(data) {
	  const node_metrics = new Map([["None", "none"]]);
	  const keys = Object.keys(data[0]);

	  if (keys.includes("ground_truth_cluster")) {
	    node_metrics.set("Ground truth cluster", "ground_truth_cluster");
	  }

	  if (keys.includes("source_dataset")) {
	    node_metrics.set("Source dataset", "source_dataset");
	  }

	  return node_metrics;
	}

	function detect_edge_colour_metrics(data) {
	  const edge_metrics = new Map();
	  const keys = Object.keys(data[0]);

	  if (keys.includes("match_probability")) {
	    edge_metrics.set("Match probability", "match_probability");
	  }

	  if (keys.includes("match_weight")) {
	    edge_metrics.set("Match weight (log2 bayes factor)", "match_weight");
	  }

	  if (keys.includes("tf_adjusted_match_prob")) {
	    edge_metrics.set("TF adjusted match probability", "tf_adjusted_match_prob");
	  }

	  if (keys.includes("edge_betweenness")) {
	    edge_metrics.set("Edge betweenness", "edge_betweenness");
	  }

	  if (keys.includes("is_bridge")) {
	    edge_metrics.set("Is bridge", "is_bridge");
	  }

	  if (keys.includes("is_false_positive")) {
	    edge_metrics.set("Is false positive", "is_false_positive");
	  }

	  return edge_metrics;
	}

	const metric_vis_args = {
	  edge_colour: {
	    match_probability: {
	      edge_metric_name: "match_probability",
	      reverse: false,
	      domain: [0, 1],
	      range: { scheme: "redyellowgreen" },
	    },
	    tf_adjusted_match_prob: {
	      edge_metric_name: "tf_adjusted_match_prob",
	      reverse: false,
	      domain: [0, 1],
	      range: { scheme: "redyellowgreen" },
	    },
	    match_weight: {
	      edge_metric_name: "match_weight",
	      reverse: false,
	      domain: [-20, 20],
	      range: { scheme: "redyellowgreen" },
	    },
	    edge_betweenness: {
	      edge_metric_name: "edge_betweenness",
	      reverse: true,
	      domain: [0, 1],
	      range: { scheme: "redyellowgreen" },
	    },
	    is_bridge: {
	      edge_metric_name: "is_bridge",
	      reverse: true,
	      domain: [0, 1],
	      range: { scheme: "redyellowgreen" },
	    },
	    is_false_positive: {
	      edge_metric_name: "is_false_positive",
	      reverse: true,
	      domain: [0, 1],
	      range: { scheme: "redyellowgreen" },
	    },
	  },
	  node_size: {
	    eigen_centrality: {
	      node_metric_name: "eigen_centrality",
	      reverse: false,
	      domain: { data: "node-data", field: "eigen_centrality" },
	      range: [100, 2000],
	    },
	  },
	  node_colour: {
	    ground_truth_cluster: {
	      node_metric_name: "ground_truth_cluster",

	      domain: { data: "node-data", field: "ground_truth_cluster" },
	      range: { scheme: "category10" },
	    },
	    source_dataset: {
	      node_metric_name: "source_dataset",

	      domain: { data: "node-data", field: "source_dataset" },
	      range: { scheme: "category10" },
	    },
	  },
	};

	function node_rows_to_table(nodes_list_of_dicts, splink_settings) {
	  const first_cols = splink_settings.cols_used_by_model_inc_add_to_retain;
	  let all_cols = Object.keys(nodes_list_of_dicts[0]);

	  all_cols = all_cols.filter(function (el) {
	    return !first_cols.includes(el);
	  });

	  let cols = first_cols.concat(all_cols);

	  let new_data = nodes_list_of_dicts.map((d) => {
	    let d2 = {};
	    cols.forEach((c) => {
	      d2[c] = d[c];
	    });
	    return d2;
	  });

	  return table(new_data, { layout: "auto" });
	}

	exports.ComparisonColumn = ComparisonColumn;
	exports.ForceDirectedChart = ForceDirectedChart;
	exports.Inspector = Inspector;
	exports.Runtime = Runtime;
	exports.SplinkSettings = SplinkSettings;
	exports.bayes_factor_to_prob = bayes_factor_to_prob;
	exports.checkbox = checkbox;
	exports.cloneDeep = cloneDeep;
	exports.comparison_column_table = comparison_column_table;
	exports.define = define;
	exports.detect_edge_colour_metrics = detect_edge_colour_metrics;
	exports.detect_node_colour_metrics = detect_node_colour_metrics;
	exports.detect_node_size_metrics = detect_node_size_metrics;
	exports.edge_row_to_table = edge_row_to_table;
	exports.filter_edges_with_cluster_id = filter_edges_with_cluster_id;
	exports.filter_nodes_with_cluster_id = filter_nodes_with_cluster_id;
	exports.format_edges_data_for_force_directed = format_edges_data_for_force_directed;
	exports.format_nodes_data_for_force_directed = format_nodes_data_for_force_directed;
	exports.get_unique_cluster_ids_from_nodes_data = get_unique_cluster_ids_from_nodes_data;
	exports.get_waterfall_chart_data = get_waterfall_chart_data;
	exports.get_waterfall_chart_spec = get_waterfall_chart_spec;
	exports.log2 = log2;
	exports.log2_bayes_factor_to_prob = log2_bayes_factor_to_prob;
	exports.metric_vis_args = metric_vis_args;
	exports.node_row_to_table = node_row_to_table;
	exports.node_rows_to_table = node_rows_to_table;
	exports.prob_to_bayes_factor = prob_to_bayes_factor;
	exports.prob_to_log2_bayes_factor = prob_to_log2_bayes_factor;
	exports.range = range;
	exports.select = select;
	exports.single_cluster_table = single_cluster_table;
	exports.table = table;

	Object.defineProperty(exports, '__esModule', { value: true });

}));

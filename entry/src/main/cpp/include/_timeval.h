// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _TIMEVAL_DEFINED
#define _TIMEVAL_DEFINED

struct timeval
{
	long tv_sec;
	long tv_usec;
};

#define timerisset(tvp)		((tvp)->tv_sec || (tvp)->tv_usec)
#define timercmp(tvp,uvp,cmp)					\
		((tvp)->tv_sec cmp (uvp)->tv_sec ||		\
		 ((tvp)->tv_sec == (uvp)->tv_sec && (tvp)->tv_usec cmp (uvp)->tv_usec))
#define timerclear(tvp)		(tvp)->tv_sec = (tvp)->tv_usec = 0

#endif /* _TIMEVAL_DEFINED */
